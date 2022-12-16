import math

import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from clrnet.utils.lane import Lane
from clrnet.models.losses.focal_loss import FocalLoss
from clrnet.models.losses.accuracy import accuracy
from clrnet.ops import nms

from clrnet.models.utils.roi_gather import ROIGather, LinearModule
from clrnet.models.utils.seg_decoder import SegDecoder
from clrnet.models.utils.dynamic_assign import assign
from clrnet.models.losses.lineiou_loss import liou_loss
from ..registry import HEADS


@HEADS.register_module
class CLRHead(nn.Module):
    def __init__(self,
                 num_points=72,
                 prior_feat_channels=64,
                 fc_hidden_dim=64,
                 num_priors=192,
                 num_fc=2,
                 refine_layers=3,
                 sample_points=36,
                 cfg=None):
        super(CLRHead, self).__init__()
        self.cfg = cfg
        self.img_w = self.cfg.img_w
        self.img_h = self.cfg.img_h
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.num_priors = num_priors
        self.sample_points = sample_points
        self.refine_layers = refine_layers
        self.fc_hidden_dim = fc_hidden_dim

        self.register_buffer(name='sample_x_indexs', tensor=(torch.linspace(
            0, 1, steps=self.sample_points, dtype=torch.float32) *
                                self.n_strips).long())
        self.register_buffer(name='prior_feat_ys', tensor=torch.flip(
            (1 - self.sample_x_indexs.float() / self.n_strips), dims=[-1]))
        self.register_buffer(name='prior_ys', tensor=torch.linspace(1,
                                       0,
                                       steps=self.n_offsets,
                                       dtype=torch.float32))

        self.prior_feat_channels = prior_feat_channels

        self._init_prior_embeddings()
        # priors: (num_priors, 6 + S)
        # priors_on_featmap: (num_priors, N_sample)
        init_priors, priors_on_featmap = self.generate_priors_from_embeddings() #None, None
        self.register_buffer(name='priors', tensor=init_priors)
        self.register_buffer(name='priors_on_featmap', tensor=priors_on_featmap)

        # generate xys for feature map
        self.seg_decoder = SegDecoder(self.img_h, self.img_w,
                                      self.cfg.num_classes,
                                      self.prior_feat_channels,
                                      self.refine_layers)

        reg_modules = list()
        cls_modules = list()
        for _ in range(num_fc):
            reg_modules += [*LinearModule(self.fc_hidden_dim)]
            cls_modules += [*LinearModule(self.fc_hidden_dim)]
        self.reg_modules = nn.ModuleList(reg_modules)
        self.cls_modules = nn.ModuleList(cls_modules)

        self.roi_gather = ROIGather(self.prior_feat_channels, self.num_priors,
                                    self.sample_points, self.fc_hidden_dim,
                                    self.refine_layers)

        self.reg_layers = nn.Linear(
            self.fc_hidden_dim, self.n_offsets + 1 + 2 +
            1)  # n offsets + 1 length + start_x + start_y + theta
        self.cls_layers = nn.Linear(self.fc_hidden_dim, 2)

        weights = torch.ones(self.cfg.num_classes)
        weights[0] = self.cfg.bg_weight
        self.criterion = torch.nn.NLLLoss(ignore_index=self.cfg.ignore_label,
                                     weight=weights)

        # init the weights here
        self.init_weights()

    # function to init layer weights
    def init_weights(self):
        # initialize heads
        for m in self.cls_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)

        for m in self.reg_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)

    def pool_prior_features(self, batch_features, num_priors, prior_xs):
        '''
        pool prior feature from feature map.
        Args:
            batch_features (Tensor): Input feature maps, shape: (B, C, H, W)
            num_priors: int
            prior_xs: (B, num_priors, N_sample)
        Returns:
            feature: (B*num_priors, C, N_sample, 1)
        '''

        batch_size = batch_features.shape[0]

        # (B, num_priors, N_sample， 1)
        prior_xs = prior_xs.view(batch_size, num_priors, -1, 1)
        # (N_sample, ) --> (B*N_sample*num_priors, ) --> (B, num_priors, N_sample, 1)
        prior_ys = self.prior_feat_ys.repeat(batch_size * num_priors).view(
            batch_size, num_priors, -1, 1)

        # (0, 1) --> (-1, 1)
        prior_xs = prior_xs * 2. - 1.
        prior_ys = prior_ys * 2. - 1.
        grid = torch.cat((prior_xs, prior_ys), dim=-1)      # (B, num_priors, N_sample, 2)
        # (B, C, num_priors, N_sample) --> (B, num_priors, C, N_sample)
        feature = F.grid_sample(batch_features, grid,
                                align_corners=True).permute(0, 2, 1, 3)

        # (B*num_priors, C, N_sample, 1)
        feature = feature.reshape(batch_size * num_priors,
                                  self.prior_feat_channels, self.sample_points,
                                  1)

        return feature

    def generate_priors_from_embeddings(self):
        """
        Returns:
            priors: (num_priors, 6+S)
            priors_on_featmap: (num_priors, N_sample)
        """
        predictions = self.prior_embeddings.weight  # (num_priors, 3)

        # 2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, 72 coordinates,
        # score[0] = negative prob, score[1] = positive prob
        # (num_priors, 6+S)
        priors = predictions.new_zeros(
            (self.num_priors, 2 + 2 + 2 + self.n_offsets), device=predictions.device)

        priors[:, 2:5] = predictions.clone()    # 1 start_y, 1 start_x, 1 theta
        # self.prior_ys: [1, ..., 0]  len=S   从底向上排列
        # start_x + ((1 - prior_y) - start_y) / tan(theta)     normalized x coords   按照图像底部-->顶部排列.
        priors[:, 6:] = (
            priors[:, 3].unsqueeze(1).clone().repeat(1, self.n_offsets) *
            (self.img_w - 1) +
            ((1 - self.prior_ys.repeat(self.num_priors, 1) -
              priors[:, 2].unsqueeze(1).clone().repeat(1, self.n_offsets)) *
             self.img_h / torch.tan(priors[:, 4].unsqueeze(1).clone().repeat(
                 1, self.n_offsets) * math.pi + 1e-5))) / (self.img_w - 1)

        # init priors on feature map
        # (num_priors, N_sample)
        priors_on_featmap = priors.clone()[..., 6 + self.sample_x_indexs]

        return priors, priors_on_featmap

    def _init_prior_embeddings(self):
        # [start_y, start_x, theta] -> all normalize
        self.prior_embeddings = nn.Embedding(self.num_priors, 3)

        bottom_priors_nums = self.num_priors * 3 // 4
        left_priors_nums, _ = self.num_priors // 8, self.num_priors // 8

        strip_size = 0.5 / (left_priors_nums // 2 - 1)
        bottom_strip_size = 1 / (bottom_priors_nums // 4 + 1)
        for i in range(left_priors_nums):
            nn.init.constant_(self.prior_embeddings.weight[i, 0],
                              (i // 2) * strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 0.)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.16 if i % 2 == 0 else 0.32)

        for i in range(left_priors_nums,
                       left_priors_nums + bottom_priors_nums):
            nn.init.constant_(self.prior_embeddings.weight[i, 0], 0.)
            nn.init.constant_(self.prior_embeddings.weight[i, 1],
                              ((i - left_priors_nums) // 4 + 1) *
                              bottom_strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.2 * (i % 4 + 1))

        for i in range(left_priors_nums + bottom_priors_nums, self.num_priors):
            nn.init.constant_(
                self.prior_embeddings.weight[i, 0],
                ((i - left_priors_nums - bottom_priors_nums) // 2) *
                strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 1.)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.68 if i % 2 == 0 else 0.84)

    # forward function here
    def forward(self, x, **kwargs):
        '''
        Take pyramid features as input to perform Cross Layer Refinement and finally output the prediction lanes.
        Each feature is a 4D tensor.
        Args:
            x: input features (list[Tensor])
            kwargs:
                dict {
                    'img': (B, 3, img_H, img_W),
                    'lane_line': (B, max_lanes, 6+S),  2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, S coordinates
                    'seg': (B, img_H, img_W),
                    'meta': List[dict0, dict1, ...]
                }
        Return:
            prediction_list: each layer's prediction result
            seg: segmentation result for auxiliary loss
        '''
        batch_features = list(x[len(x) - self.refine_layers:])
        # high level --> low level
        batch_features.reverse()    # List[(B, C, H5, W5), (B, C, H4, W4), (B, C, H3, W3)]
        batch_size = batch_features[-1].shape[0]    # batch_size

        if self.training:
            # 由于priors是可学习的，因此训练过程中一直在变化.
            # priors: (num_priors, 6+S)
            # priors_on_featmap: (num_priors, N_sample)
            self.priors, self.priors_on_featmap = self.generate_priors_from_embeddings()

        # priors: (B, num_priors, 6+S)  2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, 72 coordinates,
        # priors_on_featmap: (B, num_priors, N_sample)
        priors, priors_on_featmap = self.priors.repeat(batch_size, 1,
                                                  1), self.priors_on_featmap.repeat(
                                                      batch_size, 1, 1)

        predictions_lists = []

        # iterative refine
        prior_features_stages = []
        for stage in range(self.refine_layers):
            num_priors = priors_on_featmap.shape[1]
            prior_xs = torch.flip(priors_on_featmap, dims=[2])      # 图像顶部-->底部

            batch_prior_features = self.pool_prior_features(
                batch_features[stage], num_priors, prior_xs)    # (B*num_priors, C, N_sample, 1)
            prior_features_stages.append(batch_prior_features)

            fc_features = self.roi_gather(prior_features_stages,
                                          batch_features[stage], stage)     # (B, num_priors, C)

            # (B*num_priors, C)
            fc_features = fc_features.view(num_priors, batch_size,
                                           -1).reshape(batch_size * num_priors,
                                                       self.fc_hidden_dim)

            cls_features = fc_features.clone()
            reg_features = fc_features.clone()
            for cls_layer in self.cls_modules:
                # (B*num_priors, C)
                cls_features = cls_layer(cls_features)
            for reg_layer in self.reg_modules:
                # (B*num_priors, C)
                reg_features = reg_layer(reg_features)

            # (B*num_priors, C) --> (B*num_priors, 2)
            cls_logits = self.cls_layers(cls_features)
            # (B*num_priors, C) --> (B*num_priors, 2+1+1+n_offsets)
            reg = self.reg_layers(reg_features)

            cls_logits = cls_logits.reshape(
                batch_size, -1, cls_logits.shape[1])  # (B, num_priors, 2)
            reg = reg.reshape(batch_size, -1, reg.shape[1])    # (B, num_priors, 2+1+1+n_offsets)

            predictions = priors.clone()    # (B, num_priors, 6+S)
            predictions[:, :, :2] = cls_logits  # (B*num_priors, 2)

            # start_y, 1 start_x, 1 theta   (B, num_priors, 3)
            predictions[:, :,
                        2:5] += reg[:, :, :3]  # also reg theta angle here
            predictions[:, :, 5] = reg[:, :, 3]  # length   (B, num_priors), 这似乎是一个normalized length.

            def tran_tensor(t):
                return t.unsqueeze(2).clone().repeat(1, 1, self.n_offsets)

            # (B, num_priors, S)
            predictions[..., 6:] = (
                tran_tensor(predictions[..., 3]) * (self.img_w - 1) +
                ((1 - self.prior_ys.repeat(batch_size, num_priors, 1) -
                  tran_tensor(predictions[..., 2])) * self.img_h /
                 torch.tan(tran_tensor(predictions[..., 4]) * math.pi + 1e-5))) / (self.img_w - 1)

            # 更新后的line priors(考虑start_x、start_y、theta和length的更新)，作为下一个layer的line prior.
            prediction_lines = predictions.clone()      # (B, num_priors, 6+S)
            # (B, num_priors, S)
            predictions[..., 6:] += reg[..., 4:]

            predictions_lists.append(predictions)

            if stage != self.refine_layers - 1:
                priors = prediction_lines.detach().clone()
                priors_on_featmap = priors[..., 6 + self.sample_x_indexs]

        if self.training:
            seg = None
            # (B, refine_layers*C, H3, W3)
            seg_features = torch.cat([
                F.interpolate(feature,
                              size=[
                                  batch_features[-1].shape[2],
                                  batch_features[-1].shape[3]
                              ],
                              mode='bilinear',
                              align_corners=False)
                for feature in batch_features
            ], dim=1)
            # (B, refine_layers*C, H3, W3)  --> (B, num_class, img_H, img_W)
            seg = self.seg_decoder(seg_features)
            output = {
                'predictions_lists': predictions_lists,     # List[(B, num_priors, 6+S), (B, num_priors, 6+S), (B, num_priors, 6+S)]
                'seg': seg      # (B, num_class, img_H, img_W)
            }
            return self.loss(output, kwargs['batch'])

        return predictions_lists[-1]

    def predictions_to_pred(self, predictions):
        '''
        Convert predictions to internal Lane structure for evaluation.
        Args:
            predictions: (N_pred, 6+S)
                6+S: 2 scores, 1 start_y(normalized), 1 start_x(normalized), 1 theta(normalized), 1 length(absolute), S coordinates(normalized)

        '''
        self.prior_ys = self.prior_ys.to(predictions.device)
        self.prior_ys = self.prior_ys.double()
        lanes = []
        for lane in predictions:
            lane_xs = lane[6:]  # normalized value
            start = min(max(0, int(round(lane[2].item() * self.n_strips))),
                        self.n_strips)
            length = int(round(lane[5].item()))
            end = start + length - 1
            end = min(end, len(self.prior_ys) - 1)
            # end = label_end
            # if the prediction does not start at the bottom of the image,
            # extend its prediction until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) & (lane_xs[:start] <= 1.)
                       ).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.prior_ys[lane_xs >= 0]   # (N_valid, )   由底向上
            lane_xs = lane_xs[lane_xs >= 0]         # (N_valid, )
            lane_xs = lane_xs.flip(0).double()      # (N_valid, )   由上向底
            lane_ys = lane_ys.flip(0)               # (N_valid, )   由上向底

            lane_ys = (lane_ys * (self.cfg.ori_img_h - self.cfg.cut_height) +
                       self.cfg.cut_height) / self.cfg.ori_img_h
            if len(lane_xs) <= 1:
                continue
            points = torch.stack(
                (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)),
                dim=1).squeeze(2)      # (N_valid, 2)
            lane = Lane(points=points.cpu().numpy(),
                        metadata={
                            'start_x': lane[3],
                            'start_y': lane[2],
                            'conf': lane[1]
                        })
            lanes.append(lane)
        return lanes

    def loss(self,
             output,
             batch,
             cls_loss_weight=2.,
             xyt_loss_weight=0.5,
             iou_loss_weight=2.,
             seg_loss_weight=1.):
        """
        Args:
            output:
                dict = {
                    'predictions_lists': predictions_lists,     # List[(B, num_priors, 6+S), (B, num_priors, 6+S), (B, num_priors, 6+S)]
                        6+S: 2 scores, 1 start_y(normalized), 1 start_x(normalized), 1 theta, 1 length(normalized), 72 coordinates(normalized),
                    'seg': seg      # (B, num_class, img_H, img_W)
                }
            batch:
                dict {
                    'img': (B, 3, img_H, img_W),
                    'lane_line': (B, max_lanes, 6+S),  2 scores, 1 start_y (normalized), 1 start_x (absolute), 1 theta, 1 length (absolute), S coordinates(absolute)
                    'seg': (B, img_H, img_W),
                    'meta': List[dict0, dict1, ...]
                }
            cls_loss_weight:
            xyt_loss_weight:
            iou_loss_weight:
            seg_loss_weight:
        Returns:

        """

        if self.cfg.haskey('cls_loss_weight'):
            cls_loss_weight = self.cfg.cls_loss_weight
        if self.cfg.haskey('xyt_loss_weight'):
            xyt_loss_weight = self.cfg.xyt_loss_weight
        if self.cfg.haskey('iou_loss_weight'):
            iou_loss_weight = self.cfg.iou_loss_weight
        if self.cfg.haskey('seg_loss_weight'):
            seg_loss_weight = self.cfg.seg_loss_weight

        predictions_lists = output['predictions_lists']     # List[(B, num_priors, 6+S), (B, num_priors, 6+S), (B, num_priors, 6+S)]
        targets = batch['lane_line'].clone()    # (B, max_lanes, 6+S)
        cls_criterion = FocalLoss(alpha=0.25, gamma=2.)
        cls_loss = 0
        reg_xytl_loss = 0
        iou_loss = 0
        cls_acc = []

        cls_acc_stage = []
        for stage in range(self.refine_layers):
            predictions_list = predictions_lists[stage]     # (B, num_priors, 6+S)
            for predictions, target in zip(predictions_list, targets):
                # predictions: (num_priors, 6+S)
                # target: (max_lanes, 6+S)
                target = target[target[:, 1] == 1]      # (num_targets, 6+S)

                if len(target) == 0:
                    # If there are no targets, all predictions have to be negatives (i.e., 0 confidence)
                    cls_target = predictions.new_zeros(predictions.shape[0]).long()
                    cls_pred = predictions[:, :2]
                    cls_loss = cls_loss + cls_criterion(
                        cls_pred, cls_target).sum()
                    continue

                with torch.no_grad():
                    # matched_row_inds (Tensor): the index of assigned priors.     # (N_pos, )
                    # matched_col_inds (Tensor): the corresponding ground truth index.   # (N_pos, )
                    matched_row_inds, matched_col_inds = assign(
                        predictions, target, self.img_w, self.img_h)

                # classification targets
                cls_target = predictions.new_zeros(predictions.shape[0]).long()     # (num_priors, )
                cls_target[matched_row_inds] = 1
                cls_pred = predictions[:, :2]       # (num_priors, 2)

                # regression targets -> [start_y, start_x, theta, length] (all transformed to absolute values),
                # only on matched pairs
                reg_yxtl = predictions[matched_row_inds, 2:6]       # (N_pos, 4)
                reg_yxtl[:, 0] *= self.n_strips
                reg_yxtl[:, 1] *= (self.img_w - 1)
                reg_yxtl[:, 2] *= 180
                reg_yxtl[:, 3] *= self.n_strips

                target_yxtl = target[matched_col_inds, 2:6].clone()     # (N_pos, 4)

                # regression targets -> S coordinates (all transformed to absolute values)
                reg_pred = predictions[matched_row_inds, 6:]    # (N_pos, S)
                reg_pred *= (self.img_w - 1)
                reg_targets = target[matched_col_inds, 6:].clone()      # (N_pos, S)

                with torch.no_grad():
                    predictions_starts = torch.clamp(
                        (predictions[matched_row_inds, 2] *
                         self.n_strips).round().long(), 0,
                        self.n_strips)  # ensure the predictions starts is valid
                    target_starts = (target[matched_col_inds, 2] *
                                     self.n_strips).round().long()
                    target_yxtl[:, -1] -= (predictions_starts - target_starts
                                           )  # reg length

                # Loss calculation
                # Question: 使用的是Focal Loss, 那除以的应该是正样本的数量.
                cls_loss = cls_loss + cls_criterion(cls_pred, cls_target).sum(
                ) / target.shape[0]

                target_yxtl[:, 0] *= self.n_strips
                target_yxtl[:, 2] *= 180

                reg_xytl_loss = reg_xytl_loss + F.smooth_l1_loss(
                    reg_yxtl, target_yxtl,
                    reduction='none').mean()

                iou_loss = iou_loss + liou_loss(
                    reg_pred, reg_targets,
                    self.img_w, length=15)

                # calculate acc
                cls_accuracy = accuracy(cls_pred, cls_target)
                cls_acc_stage.append(cls_accuracy)

            cls_acc.append(sum(cls_acc_stage) / len(cls_acc_stage))

        # extra segmentation loss
        seg_loss = self.criterion(F.log_softmax(output['seg'], dim=1),
                             batch['seg'].long())

        cls_loss /= (len(targets) * self.refine_layers)
        reg_xytl_loss /= (len(targets) * self.refine_layers)
        iou_loss /= (len(targets) * self.refine_layers)

        loss = cls_loss * cls_loss_weight + reg_xytl_loss * xyt_loss_weight \
            + seg_loss * seg_loss_weight + iou_loss * iou_loss_weight

        return_value = {
            'loss': loss,
            'loss_stats': {
                'loss': loss,
                'cls_loss': cls_loss * cls_loss_weight,
                'reg_xytl_loss': reg_xytl_loss * xyt_loss_weight,
                'seg_loss': seg_loss * seg_loss_weight,
                'iou_loss': iou_loss * iou_loss_weight
            }
        }

        for i in range(self.refine_layers):
            return_value['loss_stats']['stage_{}_acc'.format(i)] = cls_acc[i]

        return return_value

    def get_lanes(self, output, as_lanes=True):
        '''
        Convert model output to lanes.
        Args:
            output: (B, num_priors, 6+S)
                6+S: 2 scores, 1 start_y(normalized), 1 start_x(normalized), 1 theta(normalized), 1 length(normalized), S coordinates(normalized)
        '''
        softmax = nn.Softmax(dim=1)

        decoded = []
        for predictions in output:
            # filter out the conf lower than conf threshold
            threshold = self.cfg.test_parameters.conf_threshold
            scores = softmax(predictions[:, :2])[:, 1]
            keep_inds = scores >= threshold
            predictions = predictions[keep_inds]
            scores = scores[keep_inds]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue

            nms_predictions = predictions.detach().clone()
            # (num_pred, 5+S)   2 scores, 1 start_y(normalized), 1 start_x(normalized), 1 length(normalized), S coordinates(normalized)
            nms_predictions = torch.cat(
                [nms_predictions[..., :4], nms_predictions[..., 5:]], dim=-1)
            # normalized length --> absolute length
            nms_predictions[..., 4] = nms_predictions[..., 4] * self.n_strips
            # normalized x_coords --> absolute x_coords
            nms_predictions[...,
                            5:] = nms_predictions[..., 5:] * (self.img_w - 1)

            keep, num_to_keep, _ = nms(
                nms_predictions,
                scores,
                overlap=self.cfg.test_parameters.nms_thres,
                top_k=self.cfg.max_lanes)
            keep = keep[:num_to_keep]       # (N_keep, )
            predictions = predictions[keep]     # (N_keep, 6+S)

            if predictions.shape[0] == 0:
                decoded.append([])
                continue

            # normalized length --> absolute length
            predictions[:, 5] = torch.round(predictions[:, 5] * self.n_strips)
            if as_lanes:
                pred = self.predictions_to_pred(predictions)
            else:
                pred = predictions
            decoded.append(pred)


        return decoded

import torch
from clrnet.models.losses.lineiou_loss import line_iou


def distance_cost(predictions, targets, img_w):
    """
    repeat predictions and targets to generate all combinations
    use the abs distance as the new distance cost
    Args:
        predictions:  (num_priors, 6+S)
        targets: (num_targets, 6+S)
            6+S: 2 scores, 1 start_y (normalized), 1 start_x (absolute), 1 theta, 1 length, S coordinates(absolute)
    Returns:
        distances: (num_priors, num_targets)
    """
    num_priors = predictions.shape[0]
    num_targets = targets.shape[0]

    # (num_priors, S) --> (num_priors, 1, S) --> (num_priors, num_targets, S)
    predictions = predictions.unsqueeze(dim=1).repeat(1, num_targets, 1)[..., 6:]
    # (num_targets, S) --> (1, num_targets, S) --> (num_priors, num_targets, S)
    targets = targets.unsqueeze(dim=0).repeat(num_priors, 1, 1)[..., 6:]

    invalid_masks = (targets < 0) | (targets >= img_w)      # (num_priors, num_targets, S)
    lengths = (~invalid_masks).sum(dim=-1)   # (num_priors, num_targets)
    distances = torch.abs((targets - predictions))      # (num_priors, num_targets, S)
    distances[invalid_masks] = 0.
    distances = distances.sum(dim=-1) / (lengths.float() + 1e-9)    # (num_priors, num_targets)
    return distances


def focal_cost(cls_pred, gt_labels, alpha=0.25, gamma=2, eps=1e-12):
    """
    Args:
        cls_pred (Tensor): Predicted classification logits, shape
            (num_priors, num_class).
        gt_labels (Tensor): Label of `gt_bboxes`, shape (num_targets,).

    Returns:
        torch.Tensor: cls_cost value  (num_priors, num_targets)
    """
    cls_pred = cls_pred.sigmoid()
    neg_cost = -(1 - cls_pred + eps).log() * (1 - alpha) * cls_pred.pow(gamma)
    pos_cost = -(cls_pred + eps).log() * alpha * (1 - cls_pred).pow(gamma)
    cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
    return cls_cost


def dynamic_k_assign(cost, pair_wise_ious):
    """
    Assign grouth truths with priors dynamically.

    Args:
        cost: the assign cost. # (num_priors, num_target)
        pair_wise_ious: iou of grouth truth and priors.  # (num_priors, num_target)

    Returns:
        prior_idx: the index of assigned prior.     # (N_pos, )
        gt_idx: the corresponding ground truth index.   # (N_pos, )
    """
    matching_matrix = torch.zeros_like(cost)
    ious_matrix = pair_wise_ious
    ious_matrix[ious_matrix < 0] = 0.
    n_candidate_k = 4
    topk_ious, _ = torch.topk(ious_matrix, n_candidate_k, dim=0)    # (n_candidate_k=4, num_target)
    # 根据line iou计算动态的top_K的数量
    dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)     # (num_target, )
    num_gt = cost.shape[1]
    for gt_idx in range(num_gt):
        _, pos_idx = torch.topk(cost[:, gt_idx],    # (num_priors, )
                                k=dynamic_ks[gt_idx].item(),
                                largest=False)
        matching_matrix[pos_idx, gt_idx] = 1.0
    del topk_ious, dynamic_ks, pos_idx

    # 此时, 每个gt_lane对应dynamic_ks个priors，但可能存在一个prior对应多个gt_lanes.
    # 接下来, 强制每个prior只对应一个gt_lane, 这个gt_lane与prior的line iou最小.
    matched_gt = matching_matrix.sum(1)     # (num_priors, )
    if (matched_gt > 1).sum() > 0:
        _, cost_argmin = torch.min(cost[matched_gt > 1, :], dim=1)
        matching_matrix[matched_gt > 1, 0] *= 0.0
        matching_matrix[matched_gt > 1, cost_argmin] = 1.0

    # (N_pos, ),  (N_pos, )
    prior_idx, gt_idx = matching_matrix.nonzero(as_tuple=True)
    return prior_idx, gt_idx


def assign(
    predictions,
    targets,
    img_w,
    img_h,
    distance_cost_weight=3.,
    cls_cost_weight=1.,
):
    '''
    computes dynamicly matching based on the cost, including cls cost and lane similarity cost
    Args:
        predictions (Tensor): predictions predicted by each stage, shape: (num_priors, 6+S)
            6+S: 2 scores, 1 start_y(normalized), 1 start_x(normalized), 1 theta, 1 length, 72 coordinates(normalized),
        targets (Tensor): lane targets, shape: (num_targets, 6+S)
            2 scores, 1 start_y (normalized), 1 start_x (absolute), 1 theta, 1 length, S coordinates(absolute)
    return:
        matched_row_inds (Tensor): the index of assigned priors.     # (N_pos, )
        matched_col_inds (Tensor): the corresponding ground truth index.   # (N_pos, )
    '''
    predictions = predictions.detach().clone()
    # 对齐 predictions 和 targets
    predictions[:, 3] *= (img_w - 1)    # normalized start_x --> absolute start_x
    predictions[:, 6:] *= (img_w - 1)   # normalized x_coods --> absolute x_coords
    targets = targets.detach().clone()

    # distances cost
    distances_score = distance_cost(predictions, targets, img_w)    # (num_priors, num_targets)
    distances_score = 1 - (distances_score / torch.max(distances_score)
                           ) + 1e-2  # normalize the distance

    # classification cost
    cls_score = focal_cost(predictions[:, :2], targets[:, 1].long())    # (num_priors, num_targets)
    num_priors = predictions.shape[0]
    num_targets = targets.shape[0]

    target_start_xys = targets[:, 2:4]      # (num_targets, 2)   2: 1 start_y (normalized), 1 start_x (absolute),
    target_start_xys[..., 0] *= (img_h - 1)     # normalized start_y --> absolute start_y
    prediction_start_xys = predictions[:, 2:4]      # (num_priors, 2)   2: 1 start_y (normalized), 1 start_x (absolute),
    prediction_start_xys[..., 0] *= (img_h - 1)     # normalized start_y --> absolute start_y

    # (num_priors, num_targets)
    start_xys_score = torch.cdist(prediction_start_xys, target_start_xys,
                                  p=2)
    start_xys_score = (1 - start_xys_score / torch.max(start_xys_score)) + 1e-2

    target_thetas = targets[:, 4].unsqueeze(-1)
    # (num_priors, num_targets)
    theta_score = torch.cdist(predictions[:, 4].unsqueeze(-1),
                              target_thetas,
                              p=1).reshape(num_priors, num_targets) * 180
    theta_score = (1 - theta_score / torch.max(theta_score)) + 1e-2

    cost = -(distances_score * start_xys_score * theta_score
             )**2 * distance_cost_weight + cls_score * cls_cost_weight

    iou = line_iou(predictions[..., 6:], targets[..., 6:], img_w, aligned=False)    # (num_priors, num_target)
    # (N_pos, ),  (N_pos, )
    matched_row_inds, matched_col_inds = dynamic_k_assign(cost, iou)

    return matched_row_inds, matched_col_inds

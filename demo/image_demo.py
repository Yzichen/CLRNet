# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from argparse import ArgumentParser
from clrnet.models.registry import build_net
from clrnet.utils.config import Config
from clrnet.datasets.process import Process
from mmcv.parallel import collate, scatter
import torch
import cv2
from clrnet.utils.visualization import imshow_lanes


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', default='demo.jpg', help='Image file')
    parser.add_argument('--config', default='../configs/clrnet/clr_resnet18_seasky.py', help='Config file')
    parser.add_argument('--checkpoint', default='epoch18_1.pth', help='Checkpoint file')
    parser.add_argument('--out-file', default='demo_vis.png', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args


def to_cuda(batch):
    for k in batch:
        if not isinstance(batch[k], torch.Tensor):
            continue
        batch[k] = batch[k].cuda()
    return batch


def inference_detector(model, imgs):
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    # device = next(model.parameters()).device  # model device
    inferecne_process = Process(cfg.inferecne_process, cfg)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            img = img[cfg.cut_height:, :, :]
        else:
            # add information into dict
            img = cv2.imread(img)
            img = img[cfg.cut_height:, :, :]

        img = img.astype(np.float32)
        img = img / 255.0
        img = img[..., [2, 1, 0]]
        data = dict(img=img)
        # build the data pipeline
        data = inferecne_process(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    data = to_cuda(data)

    # forward the model
    with torch.no_grad():
        output = model(data)
        results = model.heads.get_lanes(output)

    if not is_batch:
        return results[0]
    else:
        return results


def main(args):
    cfg = Config.fromfile(args.config)
    net = build_net(cfg)
    net = net.cuda()
    net.cfg = cfg
    pretrained_model = torch.load(args.checkpoint)
    net.load_state_dict(pretrained_model['net'], strict=True)
    net.eval()

    img = cv2.imread(args.img)
    results = inference_detector(net, img)
    lanes = [lane.to_array(cfg) for lane in results]
    imshow_lanes(img, lanes, out_file='./demo_vis.png')




if __name__ == '__main__':
    # pretrained_model = torch.load("epoch18_1.pth")
    # state_dict = pretrained_model['net']
    # new_state_dict = {}
    # for name, param in state_dict.items():
    #     new_name = name.replace('module.', '', 1)
    #     new_state_dict[new_name] = param
    # pretrained_model['net'] = new_state_dict
    # torch.save(pretrained_model, "epoch18_1.pth")

    args = parse_args()
    main(args)

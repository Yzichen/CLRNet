# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import cv2
import mmcv
from clrnet.utils.config import Config
from clrnet.models.registry import build_net
from image_demo import inference_detector
import torch
from clrnet.utils.visualization import imshow_lanes


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo')
    parser.add_argument('--video', default='demo1.mp4', help='Video file')
    parser.add_argument('--config', default='../configs/clrnet/clr_resnet18_seasky.py', help='Config file')
    parser.add_argument('--checkpoint', default='epoch18_1.pth', help='Checkpoint file')
    parser.add_argument('--out', default='demo_vis1.mp4', type=str, help='Output video file')
    parser.add_argument('--show', action='store_true', help='Show video')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.out or args.show, \
        ('Please specify at least one operation (save/show the '
         'video) with the argument "--out" or "--show"')

    cfg = Config.fromfile(args.config)
    net = build_net(cfg)
    net = net.cuda()
    net.cfg = cfg
    pretrained_model = torch.load(args.checkpoint)
    net.load_state_dict(pretrained_model['net'], strict=True)
    net.eval()


    video_reader = mmcv.VideoReader(args.video)
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))

    for frame in mmcv.track_iter_progress(video_reader):
        results = inference_detector(net, frame)
        lanes = [lane.to_array(cfg) for lane in results]
        frame = imshow_lanes(frame, lanes, return_img=True)
        if args.show:
            cv2.namedWindow('video', 0)
            mmcv.imshow(frame, 'video', args.wait_time)
        if args.out:
            video_writer.write(frame)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

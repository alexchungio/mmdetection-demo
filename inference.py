#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : inference.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/1/6 下午3:40
# @ Software   : PyCharm
#-------------------------------------------------------

from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def main():
    # parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    # parser.add_argument('config', help='Config file')
    # parser.add_argument('checkpoint', help='Checkpoint file')
    # parser.add_argument(
    #     '--device', default='cuda:0', help='Device used for inference')
    # parser.add_argument(
    #     '--score-thr', type=float, default=0.3, help='bbox score threshold')
    # args = parser.parse_args()

    config = './configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint = './checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    device = 'cuda:0'
    image = './docs/demo/demo.jpg'
    score_thr = 0.3

    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint=checkpoint, device=device)
    # test a single image
    result = inference_detector(model, image)
    # show the results
    show_result_pyplot(model, image, result, score_thr=score_thr)


if __name__ == '__main__':
    main()
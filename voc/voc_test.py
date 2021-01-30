#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : voc_test.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/1/7 下午3:14
# @ Software   : PyCharm
#-------------------------------------------------------


import os.path as osp
import mmcv
from mmcv import Config

from mmdet.datasets import build_dataset
from mmdet.apis import init_detector, inference_detector, show_result_pyplot


def test():

    config = './configs/pascal_voc/custom_retinanet_r50_fpn_1x_voc0712.py'
    cfg = Config.fromfile(config)
    checkpoint = osp.join(cfg.work_dir, 'latest.pth')
    img = mmcv.imread(osp.join('./docs/demo', 'demo_1.jpg'))

    model = init_detector(config=cfg, checkpoint=checkpoint, device='cuda:0')

    dataset = [build_dataset(cfg.data.train)]
    # Add an attribute for visualization convenience
    model.CLASSES = dataset[0].CLASSES

    model.cfg = cfg
    result = inference_detector(model, img)
    show_result_pyplot(model, img, result)


if __name__ == "__main__":

    test()
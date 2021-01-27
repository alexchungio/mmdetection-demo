#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : kitti_test.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/1/6 下午8:03
# @ Software   : PyCharm
#-------------------------------------------------------

import os.path as osp
import mmcv

from mmdet.datasets import build_dataset
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmcv import Config
from mmdet.apis import set_random_seed

from kitti_config import update_config


cfg = update_config()
# cfg = Config.fromfile('./configs/kitti/cascade_r50_fpn_1x.py')

dataset = [build_dataset(cfg.data.train)]

def test():
    checkpoint = osp.join(cfg.work_dir, 'epoch_12.pth')
    img = mmcv.imread(osp.join('./data/kitti_tiny/training/image_2', '000015.jpeg'))

    model = init_detector(config=cfg, checkpoint=checkpoint, device='cuda:0')
    print(model)
    # Add an attribute for visualization convenience
    model.CLASSES = dataset[0].CLASSES

    model.cfg = cfg
    result = inference_detector(model, img)
    show_result_pyplot(model, img, result)


if __name__ == "__main__":

    test()
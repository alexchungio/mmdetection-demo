#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : kitti_dataset.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/1/6 下午5:13
# @ Software   : PyCharm
#-------------------------------------------------------


import copy
import mmcv
import os.path as osp

import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset


data_path = '../../data/kitti_tiny'
images_path = osp.join(data_path, 'training', 'image_2')
label_path = osp.join(data_path, 'training', 'label_2')

train_index_path = osp.join(data_path, 'train.txt')
val_index_path = osp.join(data_path, 'val.txt')


def load_annotation(label_path):

    # load annotations

    lines = mmcv.list_from_file(label_path)

    content = [line.strip().split(' ') for line in lines]
    bbox_names = [x[0] for x in content]
    bboxes = [[float(info) for info in x[4:8]] for x in content]

    return bbox_names, bboxes


def visual_kitti():

    CLASSES = ('Car', 'Pedestrian', 'Cyclist', 'DontCare')
    name_index = {name: index for index, name in enumerate(CLASSES)}

    image = mmcv.imread(osp.join(images_path, '000015.jpeg'))
    bbox_names, bboxes = load_annotation(osp.join(label_path, '000015.txt'))

    gt_labels = list(map(lambda x: name_index[x], bbox_names))

    bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4)

    gt_labels = np.array(gt_labels, dtype=np.long)

    mmcv.imshow_det_bboxes(image, bboxes=bboxes, labels=gt_labels, class_names=list(CLASSES))




def main():

    visual_kitti()



if __name__ == "__main__":
    main()








#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : kitti_config.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/1/6 下午7:24
# @ Software   : PyCharm
#-------------------------------------------------------

from mmcv import Config
from mmdet.apis import set_random_seed

cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')
# cfg = Config.fromfile('./configs/kitti/cascade_r50_fpn_1x.py')
print(cfg.model.roi_head)
def update_config():
    # Modify dataset type and path
    cfg.dataset_type = 'KittiTinyDataset'
    cfg.data_root = 'data/kitti_tiny'

    cfg.data.test.type = 'KittiTinyDataset'
    cfg.data.test.data_root = 'data/kitti_tiny/'
    cfg.data.test.ann_file = 'train.txt'
    cfg.data.test.img_prefix = 'training/image_2'

    cfg.data.train.type = 'KittiTinyDataset'
    cfg.data.train.data_root = 'data/kitti_tiny/'
    cfg.data.train.ann_file = 'train.txt'
    cfg.data.train.img_prefix = 'training/image_2'

    cfg.data.val.type = 'KittiTinyDataset'
    cfg.data.val.data_root = 'data/kitti_tiny/'
    cfg.data.val.ann_file = 'val.txt'
    cfg.data.val.img_prefix = 'training/image_2'

    # modify num classes of the model in box head
    cfg.model.roi_head.bbox_head.num_classes = 3
    # cfg.model.roi_head.bbox_head[0].num_classes = 3
    # cfg.model.roi_head.bbox_head[1].num_classes = 3
    # cfg.model.roi_head.bbox_head[2].num_classes = 3

    # We can still use the pre-trained Mask RCNN model though we do not need to
    # use the mask branch
    cfg.load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

    # Set up working dir to save files and logs.
    cfg.work_dir = './outputs/kitti'

    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    cfg.optimizer.lr = 0.02 / 8
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 10

    # Change the evaluation metric since we use customized dataset.
    cfg.evaluation.metric = 'mAP'
    # We can set the evaluation interval to reduce the evaluation times
    cfg.evaluation.interval = 12
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = 12

    cfg.work_dir = './outputs'

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    return cfg

def main():

    cfg = update_config()
    print(cfg.pretty_text)

if __name__ == "__main__":
    main()
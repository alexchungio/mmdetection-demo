#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : preprocess_voc.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/1/7 上午11:28
# @ Software   : PyCharm
#-------------------------------------------------------


import time
import os
import os.path as osp
import shutil
from tqdm import tqdm


dataset_root = '/media/alex/80CA308ECA308288/alex_dataset/pascal_voc'
voc_07_trainval = osp.join(dataset_root, 'VOCtrainval_06-Nov-2007', 'VOCdevkit', 'VOC2007')
voc_07_test = osp.join(dataset_root, 'VOCtest_06-Nov-2007', 'VOCdevkit', 'VOC2007')
voc_12_trainval = osp.join(dataset_root, 'VOCtrainval_11-May-2012', 'VOCdevkit', 'VOC2012')

target_root = osp.join(dataset_root, 'VOCdevkit')
joint_voc_07 = osp.join(target_root, 'VOC2007')
joint_voc_12 = osp.join(target_root, 'VOC2012')


voc_07_main_path = os.path.join(joint_voc_07, 'ImageSets', 'Main')
voc_07_img_path = os.path.join(joint_voc_07, 'JPEGImages')
voc_07_anns_path = os.path.join(joint_voc_07, 'Annotations')

voc_12_main_path = os.path.join(joint_voc_12, 'ImageSets', 'Main')
voc_12_img_path = os.path.join(joint_voc_12, 'JPEGImages')
voc_12_anns_path = os.path.join(joint_voc_12, 'Annotations')


def main():

    shutil.rmtree(joint_voc_07, ignore_errors=True)
    shutil.rmtree(joint_voc_12, ignore_errors=True)

    for year in ['VOC2007', 'VOC2012']:
        if year == 'VOC2007':
            # Main file
            os.makedirs(voc_07_main_path)
            time.sleep(0.2)
            shutil.copyfile(osp.join(voc_07_trainval, 'ImageSets', 'Main', 'trainval.txt'),
                        osp.join(voc_07_main_path, 'trainval.txt'))
            shutil.copyfile(osp.join(voc_07_test, 'ImageSets', 'Main', 'test.txt'),
                        osp.join(voc_07_main_path, 'test.txt'))

            # Images and Annotations
            os.makedirs(voc_07_img_path)
            os.makedirs(voc_07_anns_path)
            for root in [voc_07_trainval, voc_07_test]:
                src_img_dir = os.path.join(root, 'JPEGImages')
                src_anns_dir = os.path.join(root, 'Annotations')
                pbar = tqdm(list(zip(sorted(os.listdir(src_img_dir)), sorted(os.listdir(src_anns_dir)))))
                for img_name, anns_name in pbar:
                    shutil.copyfile(osp.join(src_img_dir, img_name), osp.join(voc_07_img_path, img_name))
                    shutil.copyfile(osp.join(src_anns_dir, anns_name), osp.join(voc_07_anns_path, anns_name))

        elif year == 'VOC2012':
            os.makedirs(voc_12_main_path)
            os.makedirs(voc_12_img_path)
            os.makedirs(voc_12_anns_path)
            time.sleep(0.2) # makedir need time, otherwise will failed copy

            # Main file
            shutil.copyfile(osp.join(voc_12_trainval, 'ImageSets', 'Main', 'trainval.txt'),
                        osp.join(voc_12_main_path, 'trainval.txt'))

            # Images and Annotations
            src_img_dir = os.path.join(voc_12_trainval, 'JPEGImages')
            src_anns_dir = os.path.join(voc_12_trainval, 'Annotations')
            pbar = tqdm(list(zip(sorted(os.listdir(src_img_dir)), sorted(os.listdir(src_anns_dir)))))
            for img_name, anns_name in pbar:
                shutil.copyfile(osp.join(src_img_dir, img_name), osp.join(voc_12_img_path, img_name))
                shutil.copyfile(osp.join(src_anns_dir, anns_name), osp.join(voc_12_anns_path, anns_name))

if __name__ == "__main__":
    main()




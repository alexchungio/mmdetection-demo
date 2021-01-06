#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : env_check.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/1/6 下午4:18
# @ Software   : PyCharm
#-------------------------------------------------------

import torch
import mmdet

from mmcv.ops import get_compiling_cuda_version, get_compiler_version

def main():
    # Check Pytorch installation
    print('pytorch version: {} cuda available {}'.format(torch.__version__, torch.cuda.is_available()))
    # Check MMDetection installation
    print('mmdetection version: {}'.format(mmdet.__version__))
    # Check mmcv installation
    print('cuda version: {}'.format(get_compiling_cuda_version()))
    print('GCC version: {}'.format(get_compiler_version()))

if __name__ == "__main__":
    main()
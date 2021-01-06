# mmdetection-voc
pascal voc object detect



## install and compile
1. Create a conda virtual environment and activate it.
    ```
    conda create -n open-mmlab python=3.7 -y
    conda activate open-mmlab
   ```

2. Install PyTorch and torchvision

     ```shell script
     conda install pytorch torchvision -c pytorch
    ```

    Note: Make sure that your compilation CUDA version and runtime CUDA version match.
    You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

    `E.g` If you have CUDA 10.0 installed under `/usr/local/cuda` 
     ```shell script
     nvcc --version
    ````

    ```shell
    conda install pytorch cudatoolkit=10.0 torchvision -c pytorch
    ```

3. Install mmcv-full
    ```shell script
    pip install mmcv-full
    ```
4. Install requirements 
    ```shell script
    pip install cython && pip install -r requirements.txt
    ```
5. Install mmdetection
    ```shell script
    python setup.py develop
    ```
   
## Dataset

### download dataset
    ```
    mkdir data
    wget https://download.openmmlab.com/mmdetection/data/kitti_tiny.zip -P ./data
    unzip -o ./data/kitti_tiny.zip -d ./data/

    wget https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip -P ./data
    unzip -o ./data/devkit_object.zip -d ./data/
    ```
### check dataset
    ```
    apt-get -q install tree
    tree kitti_tiny
    
    ```
### dataset description
```shell script
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
```


## Train

### download checkpoint
    ```
     makdir checkpoint
     
     wget -c http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth \
      -O checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth

    ```


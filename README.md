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



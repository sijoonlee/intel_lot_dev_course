# Udacity Intel Edge AI for IoT Developers Nanodegree Program

# Project 1 - Deploy a People Counter App

[**WRITEUP**](https://github.com/sijoonlee/intel_lot_dev_course/blob/master/nd131-openvino-fundamentals-project-starter/WRITEUP.md)  


### Reference and how-to
1. [How to Install OpenCV3 on Ubuntu 18.04](https://linuxize.com/post/how-to-install-opencv-on-ubuntu-18-04/)  
```
sudo apt install build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-22-dev

mkdir ~/opencv_build && cd ~/opencv_build
git clone https://github.com/opencv/opencv.git  
git checkout 4.2.0

git clone https://github.com/opencv/opencv_contrib.git
git checkout 4.2.0

cd ~/opencv_build/opencv
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
    -D BUILD_EXAMPLES=ON ..

make -j4 (4 CPU core)
sudo make install
pkg-config --modversion opencv4
python3 -c "import cv2; print(cv2.__version__)"
```
2. [How to install OpenVINO on linux](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html)  

3. [How to prepare Yolov3(coco) Model for OpenVINO](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html)
```
git clone https://github.com/mystic123/tensorflow-yolo-v3.git
cd tensorflow-yolo-v3
git checkout ed60b90

Download coco.names file from the DarkNet website (https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names)

Download the yolov3.weights(https://pjreddie.com/media/files/yolov3.weights)
python3 convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3.weights

python3 /opt/intel/openvino_2020.2.120/deployment_tools/model_optimizer/mo_tf.py
--input_model /path/to/yolo_v3.pb
--tensorflow_use_custom_operations_config /opt/intel/openvino_2020.2.120/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3.json
--batch 1

```
4. [Udacity's Stater Code](https://github.com/udacity/nd131-openvino-fundamentals-project-starter)  

5. and [more references](https://github.com/arunponnusamy/object-detection-opencv)


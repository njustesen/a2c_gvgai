#!/bin/bash

#Assumes you have python 3.6 and you are on Ubuntu

#Insure Pip is installed
apt-get install python3-pip python3-dev

#Install Prerequisits
apt-get install git curl cmake libopenmpi-dev python3-dev zlib1g-dev, cython

#Install Java
apt-get install openjdk-8-jdk

#Install image processing libraries
pip3 install pillow
pip3 install opencv-python

#Install GVGAI-GYM
git clone https://github.com/rubenrtorrado/GVGAI_GYM.git
pip3 install -e ./GVGAI_GYM

#Install Baselines
git clone https://github.com/openai/baselines.git
cd baselines
pip3 install -e .
git checkout 24fe3d6576dd8f4cdd5f017805be689d6fa6be8c
cd ..

#Install A2C-Code
#git clone https://github.com/njustesen/a2c_gvgai.git
#pip3 install -e ./a2c_gvgai

sh ../lib/gvgai_generator/install.sh
#sudo npm update npm -g

#Install Tensorflow GPU
#sudo pip3 install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.8.0-cp36-cp36m-linux_x86_64.whl
pip3 install tensorflow_gpu

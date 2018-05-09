#!/bin/bash

#Assumes you have python 3.6 and you are on Ubuntu

#Insure Pip is installed
sudo apt-get install python3-pip python3-dev

#Install Prerequisits
sudo apt-get install git cmake libopenmpi-dev python3-dev zlib1g-dev

#Install Pillow
pip3 install pillow

#Install Tensorflow
#pip3 install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.8.0-cp36-cp36m-linux_x86_64.whl
pip3 install tensorflow_gpu

#Install GVGAI-GYM
git clone https://github.com/rubenrtorrado/GVGAI_GYM.git
pip3 install -e ./GVGAI_GYM

#Install Baselines
git clone https://github.com/openai/baselines.git
pip3 install -e ./baselines

#Install A2C-Code
#git clone https://github.com/njustesen/a2c_gvgai.git
#pip3 install -e ./a2c_gvgai

sh ../lib/gvgai_generator/install.sh
npm update npm -g
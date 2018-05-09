#!/bin/bash

#Install CUDA 9
sudo dpkg -i cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
export PATH=/usr/local/cuda-9.1/bin${PATH:+:${PATH}}
source .bashrc

#Install Cudnn
wget https://www.dropbox.com/s/6f7wc08lr8j1ay7/cudnn-9.1-linux-x64-v7.1.tgz?dl=0
tar -zxvf cudnn-9.1-linux-x64-v7.1.tgz
# copy libs to /usr/local/cuda folder
sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
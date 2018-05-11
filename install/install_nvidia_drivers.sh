#!/bin/bash

#Only run if drivers are missing

# update packages
sudo apt-get update
sudo apt-get upgrade
 
#Add the ppa repo for NVIDIA graphics driver
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
 
#Remove old drivers
sudo apt-get purge nvidia*

#Install the recommended driver for CUDA 9.0 (currently nvidia-384.xx)
#sudo ubuntu-drivers autoinstall
sudo apt-get nvidia-384
sudo reboot
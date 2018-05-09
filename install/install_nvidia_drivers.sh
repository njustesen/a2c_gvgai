#!/bin/bash

#Only run if drivers are missing

# update packages
sudo apt-get update
sudo apt-get upgrade
 
#Add the ppa repo for NVIDIA graphics driver
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
 
#Install the recommended driver (currently nvidia-378)
sudo ubuntu-drivers autoinstall
sudo reboot
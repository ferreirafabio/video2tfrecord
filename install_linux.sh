#!/bin/sh
pip install --user --upgrade virtualenv
mkdir ~/virtual_envs
virtualenv -p python3 --system-site-packages ~/virtual_envs/tf
source ~/virtual_envs/tf/bin/activate
pip3 install -r ./requirements.txt
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0-cp36-cp36m-linux_x86_64.whl
pip3 install --user --upgrade $TF_BINARY_URL
#!/bin/sh
pip install --user --upgrade virtualenv
mkdir ~/virtual_envs
virtualenv -p python3 --system-site-packages ~/virtual_envs/tf
source ~/virtual_envs/tf/bin/activate
pip3 install -r ./requirements.txt
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.4.0-py3-none-any.whl
pip3 install --user --upgrade $TF_BINARY_URL
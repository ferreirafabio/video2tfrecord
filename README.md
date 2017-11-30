# video2tfrecords
Easily convert RGB video data (e.g. .avi) to the TensorFlow tfrecords file format for training e.g. a NN in TensorFlow. Due to common hardware/GPU RAM limitations in Deep Learning, this implementation allows to limit the number of frames per video to be stored in the tfrecords. The code automatically chooses the frame step size s.t. there is an equal separation distribution of the individual video frames. 
Implementation supports Optical Flow (currently OpenCV's calcOpticalFlowFarneback) as an additional 4th channel and can be easily adapted in this regard. Acompanying the code, I've also added a small example with two .mp4 files from which two tfrecords batches are created (1 video per tfrecords file).

This implementation was created during a research project and grew historically. Therefore, we invite users encountering bugs to pull-request a correction.

## Requirements
Successfully tested with:
- Python 3.4 and 3.6
- tensorflow 1.4.0
- opencv-python 3.3.0.10
- numpy 1.13.3 
- virtualenv 15.1.0

## Installation
#### Python environment
It is recommended to use pip and a virtual environment for a Python installation. Therefore, differentia between the two use cases:
1. you've a environment up & running: ensure it meets requirements.txt
2. you don't have an environment set up: I've added two installation scripts (for mac and linux) that can be run with `./install_*.sh` from the console. Before running the installation, please ensure a Python 3 version, pip and virtualenv is installed on your machine.

#### OpenCV
you esentially have two options for preparing the OpenCV installation:
1. (recommended) install the statically built OpenCV binaries by using the pip wrapper package ('pip3 install opencv-python')
2. build OpenCV locally from the repository [1] (e.g. refer to StackOverflow thread under [2])




## Parameters and storage details
By adjusting the parameters at the top of the code you can control:
- input dir (containing all the video files)
- output dir (to which the tfrecords should be saved)
- resolution of the images
- video file suffix (e.g. *.avi) as RegEx
- number of frames per video that are actually stored in the tfrecord entries (can be smaller than the real number of frames)
- image color depth
- if optical flow should be added as a 4th channel
- number of videos a tfrecords file should contain



The videos are stored as features in the tfrecords. Every video instance contains the following data/information:
- feature[path] (as byte string while path being "blobs/i" with 0 <= i <=number of images per video)
- feature['height'] (while height being the image height, e.g. 128)
- feature['width'] (while width being the image width, e.g. 128)
- feature['depth'] (while depth being the image depth, e.g. 4 if optical flow used)

Additional contributors: Jonas Rothfuss (https://github.com/jonasrothfuss/)

[1] https://github.com/opencv/opencv 

[2] https://stackoverflow.com/questions/20953273/install-opencv-for-python-3-3

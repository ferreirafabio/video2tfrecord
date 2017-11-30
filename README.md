# video2tfrecords
Easily convert RGB video data (e.g. .avi) to the TensorFlow tfrecords file format for training e.g. a NN in TensorFlow. Due to common hardware/GPU RAM limitations in Deep Learning, this implementation allows to limit the number of frames per video to be stored in the tfrecords. The code automatically chooses the frame step size s.t. there is an equal separation distribution of the individual video frames. 
Implementation supports Optical Flow (currently OpenCV's calcOpticalFlowFarneback) as an additional 4th channel and can be easily adapted in this regard. 

This implementation was created during a research project and grew historically. Therefore, we invite users encountering bugs to pull-request a correction.

## Requirements
Successfully tested with:
- Python 3.4 and 3.6
- tensorflow 0.12.1
- opencv-python 3.3.0.10
- numpy 1.13.3

## Installation
### Python environment
It is recommended to use pip and a virtual environment for a Python installation:
1. Therefore, this repository includes a requirements.txt

### OpenCV
you esentially have two options for preparing the OpenCV installation:
1. (recommended) install the statically built OpenCV binaries by using the pip wrapper package ('pip3 install opencv-python')
2. build OpenCV locally from the repository (e.g. refer to [1])




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

[1] https://stackoverflow.com/questions/20953273/install-opencv-for-python-3-3

# video2tfrecords
Easily convert RGB video data (e.g. .avi) to the TensorFlow tfrecords file format with the provided 3 color channels for training e.g. a NN in TensorFlow. Due to common hardware/GPU RAM limitations in Deep Learning, this implementation allows to limit the number of frames per video that are actually stored in the tfrecords. The code automatically chooses the frame step size s.t. there is an equal separation distribution of the video images. 
Implementation supports Optical Flow (currently OpenCV's calcOpticalFlowFarneback) as an additional 4th channel. 

This implementation was created during a research project and grew historically. Therefore, we invite everybody encountering bugs to pull-request a correction.

### Requirements
- successfully tested with TensorFlow 0.12.1
- OpenCV (cv2 for python 2.7) 
- numpy

### Parameters and storage details
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

Contributors: Fábio Ferreira and Jonas Rothfuss (https://github.com/jonasrothfuss/)

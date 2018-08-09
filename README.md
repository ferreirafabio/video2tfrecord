[![Downloads](http://pepy.tech/badge/video2tfrecord)](http://pepy.tech/count/video2tfrecord)
[![Build Status](https://travis-ci.org/ferreirafabio/video2tfrecord.svg?branch=master)](https://travis-ci.org/ferreirafabio/video2tfrecord)

# Description
Easily convert RGB video data (e.g. tested with .avi and .mp4) to the TensorFlow tfrecords file format for training e.g. a NN in TensorFlow. Due to common hardware/GPU RAM limitations in Deep Learning, this implementation allows to limit the number of frames per video to be stored in the tfrecords or to simply use all video frames. The code automatically chooses the frame step size s.t. there is an equal separation distribution of the individual video frames.

The implementation offers the option to include Optical Flow (currently OpenCV's calcOpticalFlowFarneback) as an additional channel to the tfrecords data (it can be easily extended in this regard, for example, by exchanging the currently used Optical Flow algorithm with a different one). Acompanying the code, we've also added a small example with two .mp4 files from which two tfrecords batches are created (1 video per tfrecords file). To access the examples, make sure to use the GitHub repo instead of the pip package.

This implementation was created during a research project and grew historically. Therefore, we invite users encountering bugs to pull-request fixes.


# Installation
run the following command:
```
pip install video2tfrecord 
```

# Writing (video) to tfrecord
After installing the package, you execute the following exemplary command to start the video-to-tfrecord conversion:
```
from video2tfrecord import convert_videos_to_tfrecord

convert_videos_to_tfrecord(source_path, destination_path, n_videos_in_record, n_frames_per_video, "*.avi") 
```

while `n_videos_in_record` being the number of videos in one single tfrecord file, `n_frames_per_video` being the number of frames to be stored per video and `source_path` containing your .avi video files. Set `n_frames_per_video="all"` if you want all video frames to be stored in the tfrecord file (keep in mind that tfrecord can become very large).

# Reading from tfrecord
see ```test.py``` for an example


## Manual installation 
If you want to set up your installation manually, use the install scripts provided in the repository. 

The package has been successfully tested with:
- Python 3.4, 3.5 and 3.6
- tensorflow 1.5.0
- opencv-python 3.4.0.12
- numpy 1.14.0

## OpenCV troubleshooting
If you encounter issues with OpenCV (e.g. because you use a different version), you can build OpenCV locally from the repository [1] (e.g. refer to StackOverflow thread under [2]). Make sure to use the specified version as in different versions there might be changes to functions within the OpenCV framework.


# Parameters and storage details
By adjusting the parameters at the top of the code you can control:
- input dir (containing all the video files)
- output dir (to which the tfrecords should be saved)
- resolution of the images
- video file suffix (e.g. *.avi) as RegEx(!include asterisk!)
- number of frames per video that are actually stored in the tfrecord entries (can be smaller than the real number of frames)
- image color depth
- if optical flow should be added as a 4th channel
- number of videos a tfrecords file should contain


The videos are stored as features in the tfrecords. Every video instance contains the following data/information:
- feature[path] (as byte string while path being "blobs/i" with 0 <= i <=number of images per video)
- feature['height'] (while height being the image height, e.g. 128)
- feature['width'] (while width being the image width, e.g. 128)
- feature['depth'] (while depth being the image depth, e.g. 4 if optical flow used)

# Future work:
- supervised learning: allow to include a label file (e.g. .csv) that specifies the relationship \<videoid> to \<label> in each row and store label information in the records
- use compression mode in TFRecordWriter (options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP))
- improve documentation
- ~~add the option to use all video frames instead of just a subset~~ (use n_frames_per_video="all")
- ~~write small exemplary script for loading the tfrecords + meta-data into a TF QueueRunner~~ (see ```test.py```)
- replace Farneback optical flow with a more sophisticated method, say dense trajectories

Additional contributors: Jonas Rothfuss (https://github.com/jonasrothfuss/)

- [1] https://github.com/opencv/opencv
- [2] https://stackoverflow.com/questions/20953273/install-opencv-for-python-3-3

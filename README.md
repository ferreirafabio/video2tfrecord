# Description
Easily convert RGB video data (e.g. tested with .avi and .mp4) to the TensorFlow tfrecords file format for training e.g. a NN in TensorFlow. Due to common hardware/GPU RAM limitations in Deep Learning, this implementation allows to limit the number of frames per video to be stored in the tfrecords. The code automatically chooses the frame step size s.t. there is an equal separation distribution of the individual video frames. 

The implementation offers the option to include Optical Flow (currently OpenCV's calcOpticalFlowFarneback) as an additional channel to the tfrecords data. Furthermore, it can be easily extended in this regard, for example, by exchanging the currently used Optical Flow algorithm with a different one. Acompanying the code, I've also added a small example with two .mp4 files from which two tfrecords batches are created (1 video per tfrecords file). Please make sure to use the GitHub repo instead of the pip installation to access the examples.

This implementation was created during a research project and grew historically. Therefore, we invite users encountering bugs to pull-request fixes.


# Installation and start
run the following command on your terminal:
```
pip install video2tfrecord 
```

afterwards you can execute the following exemplary command to start the video-to-tfrecord conversion:

```
convert_videos_to_tfrecord(source_dir, destination_dir, n_videos, n_frames, "*.avi") 
```

while `n_videos` being the number of videos in one single tfrecord file, `n_frames` being the number of frames to be stored per video and `source_dir` containing your .avi video files.
 
If you want to set up your installation manually, use the install scripts provided. 

The package has been successfully tested with:
- Python 3.4 and 3.6
- tensorflow 1.4.0
- opencv-python 3.3.0.10
- numpy 1.13.3 

## OpenCV troubleshooting
Typically, the required OpenCV dependency will be installed during the pip installation and everything should work fine. However, if you encounter issues with OpenCV (e.g. because you use a different version), you can build OpenCV locally from the repository [1] (e.g. refer to StackOverflow thread under [2]). Make sure to use the specified version as in different versions there might be missing functions.


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
- improve documentation
- write small exemplary script for loading the tfrecords + meta-data into a TF QueueRunner
- replace Farneback optical flow with a more sophisticated method, say dense trajectories

Additional contributors: Jonas Rothfuss (https://github.com/jonasrothfuss/)

- [1] https://github.com/opencv/opencv
- [2] https://stackoverflow.com/questions/20953273/install-opencv-for-python-3-3

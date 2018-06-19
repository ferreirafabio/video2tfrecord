import os
import unittest
import tensorflow as tf
import numpy as np
from video2tfrecord import convert_videos_to_tfrecord
from tensorflow.python.platform import gfile

height = 720
width = 1280
num_depth = 4
in_path = "./example/input"
out_path = "./example/output"
n_videos_per_record = 1


class Testvideo2tfrecord(unittest.TestCase):
  def test_example1(self):
    n_frames = 5
    convert_videos_to_tfrecord(source_path=in_path, destination_path=out_path,
                               n_videos_in_record=n_videos_per_record,
                               n_frames_per_video=n_frames,
                               dense_optical_flow=True,
                               file_suffix="*.mp4")

    filenames = gfile.Glob(os.path.join(out_path, "*.tfrecords"))
    n_files = len(filenames)

    self.assertTrue(filenames)
    self.assertEqual(n_files * n_videos_per_record,
                     get_number_of_records(filenames, n_frames))

  " travis ressource exhaust, passes locally for 3.6 and 3.4"
  # def test_example2(self):
  #   n_frames = 'all'
  #   convert_videos_to_tfrecord(source_path=in_path, destination_path=out_path,
  #                              n_videos_in_record=n_videos_per_record,
  #                              n_frames_per_video=n_frames,
  #                              n_channels=num_depth, dense_optical_flow=False,
  #                              file_suffix="*.mp4")
  #
  #   filenames = gfile.Glob(os.path.join(out_path, "*.tfrecords"))
  #   n_files = len(filenames)
  #
  #   self.assertTrue(filenames)
  #   self.assertEqual(n_files * n_videos_per_record,
  #                    get_number_of_records(filenames, n_frames))


def read_and_decode(filename_queue, n_frames):
  """Creates one image sequence"""

  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  image_seq = []

  if n_frames == 'all':
    n_frames = 354  # travis kills due to too large tfrecord

  for image_count in range(n_frames):
    path = 'blob' + '/' + str(image_count)

    feature_dict = {path: tf.FixedLenFeature([], tf.string),
      'height': tf.FixedLenFeature([], tf.int64),
      'width': tf.FixedLenFeature([], tf.int64),
      'depth': tf.FixedLenFeature([], tf.int64), }

    features = tf.parse_single_example(serialized_example,
                                       features=feature_dict)

    image_buffer = tf.reshape(features[path], shape=[])
    image = tf.decode_raw(image_buffer, tf.uint8)
    image = tf.reshape(image, tf.stack([height, width, num_depth]))
    image = tf.reshape(image, [1, height, width, num_depth])
    image_seq.append(image)

  image_seq = tf.concat(image_seq, 0)

  return image_seq


def get_number_of_records(filenames, n_frames):
  """
  this function determines the number of videos available in all tfrecord files. It also checks on the correct shape of the single examples in the tfrecord
  files.
  :param filenames: a list, each entry containign a (relative) path to one tfrecord file
  :return: the number of overall videos provided in the filenames list
  """

  num_examples = 0

  if n_frames == 'all':
    n_frames_in_test_video = 354
  else:
    n_frames_in_test_video = n_frames

  # create new session to determine batch_size for validation/test data
  with tf.Session() as sess_valid:
    filename_queue_val = tf.train.string_input_producer(filenames, num_epochs=1)
    image_seq_tensor_val = read_and_decode(filename_queue_val, n_frames)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess_valid.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
      while True:
        video = sess_valid.run([image_seq_tensor_val])
        assert np.shape(video) == (1, n_frames_in_test_video, height, width,
                                   num_depth), "shape in the data differs from the expected shape"
        num_examples += 1
    except tf.errors.OutOfRangeError as e:
      coord.request_stop(e)
    finally:
      coord.request_stop()
      coord.join(threads)

  return num_examples


if __name__ == '__main__':
  unittest.main()

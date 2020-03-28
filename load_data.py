""" Script containing numerous functions needed to load data into the model."""

# Copyright (c) 2018 by BQ. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def create_bottleneck_iterator(data, img_size):
    """
    Creates a Tensorflow iterator used for the bottleneck inferences.
        Args:
            data: txt file path with the corresponding data.
            img_size: image size.
        Return:
            iterator: one shot iterator with the images and labels.
    """

    global image_size
    image_size = img_size
    paths = _load_all_images_paths(data)

    with tf.variable_scope('Iterator'):
        dataset = tf.data.Dataset.from_tensor_slices(paths)
        dataset = dataset.map(_parse_data)
        dataset = dataset.batch(1)

        iterator = dataset.make_one_shot_iterator()

    return iterator


def _load_all_images_paths(data_file):
    """
    Load all image paths from a txt file and returns and np array of them.
        Args:
            data_file: txt file path with the corresponding data.
        Return:
            np.array(paths): array with the image paths.
    """
    with open(data_file, 'r') as f:
        lines = f.readlines()
        paths = []

        for i, line in enumerate(lines):
            paths.append(line.strip('\n'))
    return np.array(paths)


# Read the image of a file path, and convert it into a Tensorflow input with some modifications.
def _parse_data(path):
    """
    Load an image and transform it to an array.
        Args:
            path: txt file path with the corresponding data.
            img_size: size in pixels of the image height and width.
        Return:
            path: input file path of the image to operate.
            img: array of the image input, decoded and resized it.
    """
    img = tf.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize_images(img, (image_size, image_size))
    img = tf.image.per_image_standardization(img)

    return img, path


def create_iterator_for_diff(tfrecord_file, is_training, batch_size=64, f_lenght=1000):
    """
    Creates a one shot iterator from the TFRecord files.
        Args:
            tfrecord_file: a Tensorflow record file path with bottlenecks and labels.
            is_training: bool variable, change the dataset in order to make a train or eval iterator.
            batch_size: number of inputs per batch, 64 by default.
            f_lenght: feature lenght of the bottlenecks. Depends of the model used
        Return:
            iterator: one shot iterator.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    global feature_lenght
    feature_lenght = f_lenght
    if is_training:
        dataset = dataset.map(_parse)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=2560)

    else:
        dataset = dataset.map(_parse)

    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()

    return iterator


def _parse(serialized):
    """
    Convert the images and labels from records feature to Tensors.
        Args:
            serialized: A dataset comprising records from one TFRecord file.
        Return:
            bottleneck pair and label tensors.
    """
    # Define a dict with the data-names and types we expect to find in the TFRecords file.

    feature = {
        'bottleneck_1': tf.FixedLenFeature((feature_lenght,), tf.float32),
        'bottleneck_2': tf.FixedLenFeature((feature_lenght,), tf.float32),
        'label': tf.FixedLenFeature([], tf.int64),
    }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized, features=feature)

    # Get the image as raw bytes, and  the height, width and label as int.
    bottleneck_1 = tf.cast(parsed_example['bottleneck_1'], tf.float32)
    bottleneck_2 = tf.cast(parsed_example['bottleneck_2'], tf.float32)
    label = tf.cast(parsed_example['label'], tf.int64)

    # The image and label are now correct TensorFlow types.
    return bottleneck_1, bottleneck_2, label

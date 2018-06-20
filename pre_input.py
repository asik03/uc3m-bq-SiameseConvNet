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

import os
import random
import tensorflow as tf
import numpy as np

filtered_data_dir = './data/datasets/filtered_dataset'
imgs_paths_txt_path = "./data/all_img_paths.txt"
diff_dataset_txt_path = "./data/diff_dataset.txt"
bottlenecks_dir = "./data/bottlenecks/"

# It will generate n tuples with label "0" and other n tuples with label "1" for each class or person.
num_tuples_per_class = 30


def generate_txt_with_all_images(data_dir, path):
    with open(path, 'w') as out:
        for person in os.listdir(data_dir):
            path = os.path.join(data_dir, person)

            for img in os.listdir(path):
                img_path = os.path.join(path, img)
                out.write(img_path + '\n')
                print(img_path)


def generate_tfrecord_files(dataset, save_file):
    """ Creates the tfrecord files from a dataset file.

        Args:
            dataset: txt file with lines having 'path_to_the_image label'.
            save_file: file where the TFRecord is going to be saved.
    """
    if os.path.exists(save_file):
        print("TFRecord file already exists in", save_file)
        return

    print("Creating TFRecord file...")

    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(save_file) as writer:

        for entry in open(dataset):
            tf_example = _create_tf_example(entry)
            writer.write(tf_example.SerializeToString())

    print("TFRecord file created at", save_file)


def _create_tf_example(entry):
    """ Creates a tf.train.Example to be saved in the TFRecord file.

        Args:
            entry: string containing the path to a image and its label.
        Return:
            tf_example: tf.train.Example containing the info stored in feature
    """
    image_path_1, image_path_2, label = _get_image_and_label_from_entry(entry)

    bottleneck_1 = np.load(image_path_1)
    bottleneck_2 = np.load(image_path_2)

    # Data which is going to be stored in the TFRecord file
    feature = {
        'bottleneck_1': tf.train.Feature(float_list=tf.train.FloatList(value=bottleneck_1.reshape(-1))),
        'bottleneck_2': tf.train.Feature(float_list=tf.train.FloatList(value=bottleneck_2.reshape(-1))),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

    return tf_example


def _get_image_and_label_from_entry(entry):
    """ Get the image's path and its label from a dataset entry.

        Args:
            entry: string containing the path to a image and its label.
        Return:
            file_path: string with the path where a image is stored.
            label: int representing the class of the image
    """
    file_path_1, file_path_2, label = entry.split(" ")

    label = label.strip('\n')

    return file_path_1, file_path_2, int(label)


def create_diff_dataset_txt(diff_dataset_txt_path, bottlenecks_dir, num_tuples_per_class):
    with open(diff_dataset_txt_path, 'w') as out:
        for class_name in os.listdir(bottlenecks_dir):
            class_path = os.path.join(bottlenecks_dir, class_name)
            print(os.listdir(class_path))

            for i in range(num_tuples_per_class):
                npy_path_1 = get_random_item_from_class(class_path)
                npy_path_2 = get_random_item_from_class(class_path)

                out.write(npy_path_1 + ' ' + npy_path_2 + ' 1' + '\n')
                print(npy_path_1, npy_path_2, '1')

            for i in range(num_tuples_per_class):
                npy_path_1 = get_random_item_from_class(class_path)

                comparing_class = get_random_class(bottlenecks_dir, exclude=os.listdir(bottlenecks_dir).index(class_name))
                npy_path_2 = get_random_item_from_class(os.path.join(bottlenecks_dir, comparing_class))

                out.write(npy_path_1 + ' ' + npy_path_2 + ' 0' + '\n')
                print(npy_path_1, npy_path_2, '0')


def get_random_item_from_class(class_path):
    rand = int(random.random() * len(os.listdir(class_path)))
    npy_path = os.listdir(class_path)[rand]

    return os.path.join(class_path, npy_path)


def get_random_class(basename_dir, exclude=None):
    rand = int(random.random() * len(os.listdir(basename_dir)))

    while rand == exclude:
        rand = int(random.random() * len(os.listdir(basename_dir)))

    class_path = os.listdir(basename_dir)[rand]

    return class_path


def main():
    generate_txt_with_all_images(filtered_data_dir, imgs_paths_txt_path)

    '''Create bottlenecks with "inferece_bottlecks.py first"'''

    create_diff_dataset_txt(diff_dataset_txt_path, bottlenecks_dir, num_tuples_per_class)

    generate_tfrecord_files(diff_dataset_txt_path, "./data/tfrecord_train_file")

    generate_tfrecord_files(diff_dataset_txt_path, "./data/tfrecord_eval_file")


if __name__ == '__main__':
    main()




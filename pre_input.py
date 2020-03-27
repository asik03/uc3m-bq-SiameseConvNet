""" This script contain numerous functions that prepares the dataset before the model training."""
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

# It will generate n tuples with label "0" and other n tuples with label "1" for each class or person.
num_tuples_per_class = 30


def generate_txt_with_all_images(data_dir, path):
    """
    Generates all image paths from a directory into a text file.
        Args:
            data_dir: dataset directory path, with directory classes and their corresponding images.
            path: text file path where the image paths are going to be saved.
    """
    with open(path, "w") as out:
        for person in os.listdir(data_dir):
            path = os.path.join(data_dir, person)

            for img in os.listdir(path):
                img_path = os.path.join(path, img)
                out.write(img_path + "\n")


def create_diff_dataset_txt(bottlenecks_dir, diff_dataset_txt_path, num_tuples_per_class=3):
    """
    Creates a text file with tuples of bottleneck path pairs with a label each line. The label will be a "0" if this
    image pairs are not the same person, and 1 if they are the same.

        Args:
            bottlenecks_dir: directory where to load and shuffle the image bottlenecks
            diff_dataset_txt_path: file path where to save the tuples.
            num_tuples_per_class: number of tuples it will be generated for each class. The total number for each class
            will be 2*n.
    """
    with open(diff_dataset_txt_path, "w") as out:
        print("Creating diff dataset.")
        for class_name in os.listdir(bottlenecks_dir):
            class_path = os.path.join(bottlenecks_dir, class_name)

            for i in range(num_tuples_per_class):
                npy_path_1 = get_random_item_from_class(class_path)
                npy_path_2 = get_random_item_from_class(class_path)

                out.write(npy_path_1 + " " + npy_path_2 + " 1" + "\n")

            for i in range(num_tuples_per_class):
                npy_path_1 = get_random_item_from_class(class_path)

                comparing_class = get_random_class(bottlenecks_dir,
                                                   exclude=os.listdir(bottlenecks_dir).index(class_name))
                npy_path_2 = get_random_item_from_class(os.path.join(bottlenecks_dir, comparing_class))

                out.write(npy_path_1 + " " + npy_path_2 + " 0" + "\n")
        print("Diff dataset created")


def get_random_item_from_class(class_path):
    """
    Receives a class directory path from the dataset and returns a random file path from this directory.

        Args:
            class_path: class directory path.
        Return:
            item: random file path is going to be returned.
    """
    rand = int(random.random() * len(os.listdir(class_path)))
    npy_path = os.listdir(class_path)[rand]
    item = os.path.join(class_path, npy_path)

    return item


def get_random_class(basename_dir, exclude=None):
    """
    Receives the dataset directory path and returns a random class path from this directory.

        Args:
            basename_dir: dataset path.
        Return:
            class_path: random class path is going to be returned.
    """
    rand = int(random.random() * len(os.listdir(basename_dir)))

    while rand == exclude:
        rand = int(random.random() * len(os.listdir(basename_dir)))

    class_path = os.listdir(basename_dir)[rand]

    return class_path


def generate_tfrecord_files(dataset, save_file):
    """
    Creates the tfrecord files from a dataset file.

        Args:
            dataset: txt file with lines having tuples with pairs of image paths and the corresponding label .
            save_file: file path where the TFRecord is going to be saved.
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
    """
    Creates a tf.train.Example to be saved in the TFRecord file.

        Args:
            entry: string containing the paths of the two images and its label.
        Return:
            tf_example: tf.train.Example containing the info stored in feature
    """
    image_path_1, image_path_2, label = _get_image_and_label_from_entry(entry)

    bottleneck_1 = np.load(image_path_1)
    bottleneck_2 = np.load(image_path_2)

    # Data which is going to be stored in the TFRecord file
    feature = {
        "bottleneck_1": tf.train.Feature(float_list=tf.train.FloatList(value=bottleneck_1.reshape(-1))),
        "bottleneck_2": tf.train.Feature(float_list=tf.train.FloatList(value=bottleneck_2.reshape(-1))),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

    return tf_example


def _get_image_and_label_from_entry(entry):
    """
    Get the image"s path and its label from a dataset entry.

        Args:
            entry: string containing the path of the two images and its label.
        Return:
            file_path(1 and 2): string with the path where a image is stored.
            label: int representing the label.
    """
    file_path_1, file_path_2, label = entry.split(" ")

    label = label.strip("\n")

    return file_path_1, file_path_2, int(label)


def main(model_name=None):
    model_dir = "./data/" + model_name + "/"

    filtered_train_data_dir = "./data/datasets/filtered_train_dataset"
    filtered_eval_data_dir = "./data/datasets/filtered_eval_dataset"

    img_paths_txt_train_path = model_dir + "all_img_train_paths.txt"
    img_paths_txt_eval_path = model_dir + "all_img_eval_paths.txt"

    diff_dataset_txt_train_path = model_dir + "diff_train_dataset.txt"
    diff_dataset_txt_eval_path = model_dir + "diff_eval_dataset.txt"

    bottlenecks_train_dir = model_dir + "train_bottlenecks/"
    bottlenecks_eval_dir = model_dir + "eval_bottlenecks/"

    tfrecord_train_file_path = model_dir + "tfrecord_train_file"
    tfrecord_test_file_path = model_dir + "tfrecord_test_file"

    """Preparing data"""
    generate_txt_with_all_images(filtered_train_data_dir, img_paths_txt_train_path)
    generate_txt_with_all_images(filtered_eval_data_dir, img_paths_txt_eval_path)

    """Create bottlenecks with 'inference_bottlenecks.py' first"""
    import inference_bottlenecks
    inference_bottlenecks.inference_bottlenecks(img_paths_txt_train_path, bottlenecks_train_dir, model_name)
    inference_bottlenecks.inference_bottlenecks(img_paths_txt_eval_path, bottlenecks_eval_dir, model_name)

    create_diff_dataset_txt(bottlenecks_train_dir, diff_dataset_txt_train_path, num_tuples_per_class)
    create_diff_dataset_txt(bottlenecks_eval_dir, diff_dataset_txt_eval_path, num_tuples_per_class=3)

    generate_tfrecord_files(diff_dataset_txt_train_path, tfrecord_train_file_path)
    generate_tfrecord_files(diff_dataset_txt_eval_path, tfrecord_test_file_path)

    print("Pre_input done...")


if __name__ == "__main__":
    main()

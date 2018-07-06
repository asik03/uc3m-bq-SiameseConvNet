"""This script will create and prepare the dataset before it is used in the model"""

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

from shutil import copyfile

dataset_path = './data/datasets/lfw_mtcnnpy_182'  # Dataset obtained from LFW dataset, preprocessed with faceNet.
min_num_images_per_class = 15
max_num_images_per_class = 75
txt_with_new_dataset_train_paths = './data/datasets/filtered_dataset_train_paths'
txt_with_new_dataset_eval_paths = './data/datasets/filtered_dataset_eval_paths'
filtered_dataset_train_path = './data/datasets/filtered_train_dataset'
filtered_dataset_eval_path = './data/datasets/filtered_eval_dataset'


def _create_filter_paths_txt(dataset_path, train_path, eval_path, min_num_imgs, max_num_imgs):
    """
       Generates all filtered image paths from a directory into a text file, with the params we introduced before.
           Args:
               dataset_path: dataset directory path, with directory classes and their corresponding images.
                Its been preprocessed with facenet repository.
               train_path: text file path where the training image paths are going to be saved.
               eval_path: text file path where the eval image paths are going to be saved.
               min_num_imgs: number of imgs that each class need to have to consider to the new training dataset.
               max_num_imgs: maximum number of imgs that is going to be stored for each class.
    """
    with open(train_path, 'w') as out:
        for person in os.listdir(dataset_path):
            dir_path = os.path.join(dataset_path, person)
            if len(os.listdir(dir_path)) >= min_num_imgs:

                for i, img in enumerate(os.listdir(dir_path)):
                    if i >= max_num_imgs:
                        break
                    print(i)
                    img_path = os.path.join(dir_path, img)
                    out.write(img_path + '\n')
    with open(eval_path, 'w') as out:
        for person in os.listdir(dataset_path):
            dir_path = os.path.join(dataset_path, person)
            if len(os.listdir(dir_path)) < min_num_imgs:

                for i, img in enumerate(os.listdir(dir_path)):
                    print(i)
                    img_path = os.path.join(dir_path, img)
                    out.write(img_path + '\n')


def _create_filtered_dataset(txt_train_path, txt_eval_path, filtered_dataset_train_path, filtered_dataset_eval_path):
    """
       Creates the new filtered dataset based on a text file with the new image paths.
           Args:
               txt_train_path: text file path with all the new training image paths.
               txt_eval_path: text file path with all the new eval image paths.
               filtered_dataset_train_path: directory path where the new filtered  train dataset is going to be saved.
               filtered_dataset_eval_path: directory path where the new filtered  eval dataset is going to be saved.
    """
    with open(txt_train_path, 'r') as file:
        if not os.path.exists(filtered_dataset_train_path):
            os.mkdir(filtered_dataset_train_path)

        lines = file.readlines()

        for i, line in enumerate(lines):
            line = line.replace("\n", "")
            basedir, file = line.split(dataset_path)
            new_path = filtered_dataset_train_path + file
            dir_path = os.path.dirname(new_path)

            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

            copyfile(line, new_path)

    with open(txt_eval_path, 'r') as file:
        if not os.path.exists(filtered_dataset_eval_path):
            os.mkdir(filtered_dataset_eval_path)

        lines = file.readlines()

        for i, line in enumerate(lines):
            line = line.replace("\n", "")
            basedir, file = line.split(dataset_path)
            new_path = filtered_dataset_eval_path + file
            dir_path = os.path.dirname(new_path)

            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

            copyfile(line, new_path)


def main():
    #create_filter_paths_txt(dataset_path, txt_with_new_dataset_train_paths, txt_with_new_dataset_eval_paths,
    #                        min_num_images_per_class, max_num_images_per_class)
    #create_filtered_dataset(txt_with_new_dataset_train_paths, txt_with_new_dataset_eval_paths,
    #                        filtered_dataset_train_path, filtered_dataset_eval_path)
    pass


if __name__ == '__main__':
    main()

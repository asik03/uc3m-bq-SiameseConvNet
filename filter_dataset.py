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

dataset_path = './data/datasets/lfw_mtcnnpy_182'
min_num_images_per_class = 15
max_num_images_per_class = 75
txt_with_new_dataset_paths = './data/datasets/filtered_dataset_paths'
filtered_dataset_path = './data/datasets/filtered_dataset'


def create_filter_paths_txt(dataset_path):
    with open(txt_with_new_dataset_paths, 'w') as out:
        for person in os.listdir(dataset_path):
            dir_path = os.path.join(dataset_path, person)
            if len(os.listdir(dir_path)) >= min_num_images_per_class:
                print(" ")
                print(dir_path)
                print(len(os.listdir(dir_path)))

                for i, img in enumerate(os.listdir(dir_path)):
                    if i >= max_num_images_per_class:
                        break
                    print(i)
                    img_path = os.path.join(dir_path, img)
                    out.write(img_path + '\n')


def create_filtered_dataset(txt_with_new_dataset_paths, filtered_dataset_path):
    with open(txt_with_new_dataset_paths, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            print("Filteres dataset: ", filtered_dataset_path)
            print("Line: ", line)

            line = line.replace("\n", "")
            print(line)

            basedir, file = line.split(dataset_path)
            print("File: ", file)

            new_path = filtered_dataset_path + file

            print("New path: ", new_path)
            dir_path = os.path.dirname(new_path)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            print("Dir path", dir_path)
            copyfile(line, new_path)


def main():
    #create_filter_paths_txt(dataset)
    #create_filtered_dataset(txt_with_new_dataset_paths, filtered_dataset_path)
    pass


if __name__ == '__main__':
    main()
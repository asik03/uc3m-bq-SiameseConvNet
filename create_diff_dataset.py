'''Create a '''

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

imgs_paths_txt_path = "./data/all_img_paths.txt"
diff_dataset_txt_path = "./data/diff_dataset.txt"
bottlenecks_dir = "./data/bottlenecks/"
num_tuples_per_class = 30


def main():
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


if __name__ == '__main__':
    main()

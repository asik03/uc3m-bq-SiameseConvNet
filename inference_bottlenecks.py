
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
import tensorflow as tf
import numpy as np
import load_data as data

from model.inception_resnet_v1 import resnet_bottleneck

bottlenecks_train_dir = "./data/train_bottlenecks/"
bottlenecks_eval_dir = "./data/eval_bottlenecks/"

img_paths_txt_train_path = "./data/all_img_train_paths.txt"
img_paths_txt_eval_path = "./data/all_img_eval_paths.txt"


def inference_bottlenecks(imgs_path, dir_bottlenecks):
    with tf.Graph().as_default():

        iterator = data.create_bottleneck_iterator(imgs_path)
        img, path_tensor = iterator.get_next()

        bottleneck_tensor, end_points = resnet_bottleneck(img, phase_train=False)

        if not os.path.exists(dir_bottlenecks):
            os.mkdir(dir_bottlenecks)

        # Initializers
        init_global = tf.initializers.global_variables()
        init_local = tf.initializers.local_variables()

        # chkp.print_tensors_in_checkpoint_file('./data/weights/model-20180408-102900.ckpt-90', tensor_name='',
        # all_tensors=False, all_tensor_names=True)

        # Create a saver
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

            sess.run(init_global)
            sess.run(init_local)

            # Restore the pretrained model from FaceNet
            saver.restore(sess, './data/weights/model-20180408-102900.ckpt-90')

            for i in range(0, 100000):
                try:
                    bottleneck, path = sess.run([bottleneck_tensor, path_tensor])
                    path = path[0].decode("utf-8")

                    basename = os.path.basename(path).strip('.png')
                    class_name = os.path.basename(os.path.dirname(path))
                    new_path = os.path.join(class_name, basename)
                    new_path = os.path.join(dir_bottlenecks, new_path)

                    if not os.path.exists(os.path.split(new_path)[0]):
                        os.mkdir(os.path.split(new_path)[0])

                    np.save(new_path, bottleneck)
                    print("Bottleneck", i, "saved at:", new_path)

                except tf.errors.OutOfRangeError:
                    print('Finished.')
                    break


if __name__ == '__main__':
    inference_bottlenecks(img_paths_txt_train_path, bottlenecks_train_dir)
    inference_bottlenecks(img_paths_txt_eval_path, bottlenecks_eval_dir)

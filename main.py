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

model = "mobilenetv2"
model_dir = "./data/" + model + "/"

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

# It will generate n tuples with label "0" and other n tuples with label "1" for each class or person.
num_tuples_per_class = 30


def main():
    """Preparing data"""
    import pre_input
    pre_input.generate_txt_with_all_images(filtered_train_data_dir, img_paths_txt_train_path)
    pre_input.generate_txt_with_all_images(filtered_eval_data_dir, img_paths_txt_eval_path)

    """Create bottlenecks with 'inference_bottlecks.py' first"""
    import inference_bottlenecks
    inference_bottlenecks.inference_bottlenecks(img_paths_txt_train_path, bottlenecks_train_dir, model)
    inference_bottlenecks.inference_bottlenecks(img_paths_txt_eval_path, bottlenecks_eval_dir, model)

    pre_input.create_diff_dataset_txt(bottlenecks_train_dir, diff_dataset_txt_train_path, num_tuples_per_class)
    pre_input.create_diff_dataset_txt(bottlenecks_eval_dir, diff_dataset_txt_eval_path, num_tuples_per_class=3)

    pre_input.generate_tfrecord_files(diff_dataset_txt_train_path, tfrecord_train_file_path)
    pre_input.generate_tfrecord_files(diff_dataset_txt_eval_path, tfrecord_test_file_path)

    """Training step"""
    import train
    train.train(model)

    """Eval step"""
    import eval
    eval.evaluate(model)


if __name__ == "__main__":
    main()

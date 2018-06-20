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

import tensorflow as tf
import logging
import load_data as data

from model import inception_resnet_v1 as model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', './data/logs/', """Directory where to write event logs. """)
tf.app.flags.DEFINE_string('ckpt_dir', './data/saves/', """Directory where to save and load the checkpoints. """)
tf.app.flags.DEFINE_string('tfrecord_file', './data/tfrecord_eval_file', """File with the dataset to train. """)
tf.app.flags.DEFINE_string('batch_size', '1', """Batch size""")

dropout_keep_prob = 0.95
num_classes = 2
batch_size = 1


def eval():

    with tf.Graph().as_default():
        iterator = data.create_iterator_for_diff(FLAGS.tfrecord_file, is_training=False)
        print(iterator)

        bottlenecks_1_batch, bottlenecks_2_batch, labels_batch = iterator.get_next()
        print("bottleneck_1:", bottlenecks_1_batch)
        print("bottleneck_2:", bottlenecks_2_batch)
        print("labels_batch:", labels_batch)

        diff_bottlenecks_tensor = tf.abs(tf.subtract(bottlenecks_1_batch, bottlenecks_2_batch))
        print("Difference: ", diff_bottlenecks_tensor)

        logits = model.classify_bottlenecks(diff_bottlenecks_tensor, dropout_keep_prob, num_classes=num_classes)
        print(logits)
        # Get the class with the highest score
        predictions = tf.nn.top_k(logits, k=1)

        saver = tf.train.Saver(tf.global_variables('classify'))

        init = tf.global_variables_initializer()

        logger = init_logger()
        logger.info("Eval starts...")

        with tf.Session() as sess:
            sess.run(init)

            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))

            success = 0
            total = 0
            exec_next_step = True
            while exec_next_step is True:
                try:
                    total += 1
                    predicted, bottleneck_1, bottleneck_2, label = sess.run([predictions, bottlenecks_1_batch, bottlenecks_2_batch, labels_batch])

                    if predicted.indices[0][0] == label[0]:
                        success += 1
                    logger.info('Success rate: %.2f of %i examples', success / total * 100, total)
                except tf.errors.OutOfRangeError:
                    logger.info("Eval ends...")
                    exec_next_step = False


def main(argv=None):
    eval()


def init_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


if __name__ == "__main__":
    tf.app.run()
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

dropout_keep_prob = 0.8
num_classes = 2
batch_size = 1
success_constraint = 0.9999


def eval():

    with tf.Graph().as_default():
        # Create dataset iterator of batch 1, obtaining the statistics correctly
        iterator = data.create_iterator_for_diff(FLAGS.tfrecord_file, is_training=False, batch_size=batch_size)
        bottlenecks_1_batch, bottlenecks_2_batch, labels_batch = iterator.get_next()

        # Get the bottleneck distances between each of them, and transform to positive their values.
        diff_bottlenecks_tensor = tf.abs(tf.subtract(bottlenecks_1_batch, bottlenecks_2_batch))

        # Get the logits from the classify model.
        logits, pre_softmax = model.classify_bottlenecks(diff_bottlenecks_tensor, dropout_keep_prob, num_classes=num_classes, is_training=False)

        # Get the class with the highest score
        predictions = tf.nn.top_k(logits, k=1)

        # Create the server to load the model weights and bias.
        saver = tf.train.Saver(tf.global_variables('classify'))

        # Variables initialisation
        init = tf.global_variables_initializer()

        logger = init_logger()
        logger.info("Eval starts...")

        with tf.Session() as sess:
            sess.run(init)

            # Restoring the classifier model
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))

            # Auxiliary variables
            predicted_label = 0
            false_positives = 0
            false_negatives = 0
            true_success = 0
            false_success = 0
            exec_next_step = True

            while exec_next_step is True:
                try:
                    # Obtain the prediction(s) and the label(s) of each step.
                    predicted, label, real_value = sess.run([predictions, labels_batch, pre_softmax])

                    # Using auxiliary parameter to check the results later.
                    if predicted.indices[0][0] == 0:
                        predicted_label = 0
                    elif predicted.indices[0][0] == 1:
                        if predicted.values[0][0] >= success_constraint:
                            predicted_label = 1
                        else:
                            predicted_label = 0

                    # Obtaining confusion matrix values
                    if predicted_label == label[0] and predicted_label == 1:
                        true_success += 1

                    if predicted_label == label[0] and predicted_label == 0:
                        false_success += 1

                    if predicted_label != label[0] and predicted_label == 1:
                        false_positives += 1

                    if predicted_label != label[0] and predicted_label == 0:
                        false_negatives += 1

                except tf.errors.OutOfRangeError:
                    logger.info("Eval ends...")
                    logger.info("Total false positives: %i", false_positives)
                    logger.info("Total false negatives: %i", false_negatives)
                    logger.info("Total true success: %i", true_success)
                    logger.info("Total false success: %i", false_success)
                    logger.info("Total: %i", false_positives + false_negatives + true_success + false_success)
                    exec_next_step = False


def main(argv=None):
    eval()


def init_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


if __name__ == "__main__":
    tf.app.run()

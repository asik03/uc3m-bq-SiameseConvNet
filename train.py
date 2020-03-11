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
import logging
import load_data as data


from model import inception_resnet_v1 as model
from datetime import datetime

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('log_dir', './data/logs/logs_31_mk_16/', """Directory where to write event logs. """)
tf.app.flags.DEFINE_integer('max_steps', 4000, """Number of epochs to run.""")
tf.app.flags.DEFINE_string('save_dir', './data/saves/saves_31_mk_16/', """Directory where to save and load the checkpoints. """)
tf.app.flags.DEFINE_string('tfrecord_file', './data/tfrecord_train_file', """File with the dataset to train. """)


# Hiperparameters for the training step
num_classes = 2             # Number of neurons in the final layer of the net.
dropout_keep_prob = 0.85    # Estimated proportion of neurons to be kept from the dropout. Dropout equals 1 - dropout_keep_prob.
learning_rate = 0.001       # Learning rate of the optimizer.
batch_size = 16             # Number of elements of input on each "round".
seed = 31                   # Value used to set a random fixed value to the random variables.


def init_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


def train():
    if not os.path.exists(FLAGS.save_dir):
        if not os.path.exists('./data/saves/'):
            os.mkdir('./data/saves/')
        os.mkdir(FLAGS.save_dir)
    with tf.Graph().as_default() as g:
        global_step = tf.train.get_or_create_global_step()
        n = tf.placeholder(tf.float32)
        tf.set_random_seed(seed)

        # Get the bottlenecks and labels from the dataset using the iterator
        iterator = data.create_iterator_for_diff(FLAGS.tfrecord_file, is_training=True, batch_size=batch_size)
        bottlenecks_1_batch, bottlenecks_2_batch, labels_batch = iterator.get_next()

        # Get the absolute difference bottlenecks, using tensorflow functions.
        diff_bottlenecks_tensor = tf.abs(tf.subtract(bottlenecks_1_batch, bottlenecks_2_batch))

        # Obtain the logits from the bottlenecks difference.
        logits = model.classify_bottlenecks(diff_bottlenecks_tensor, dropout_keep_prob=dropout_keep_prob, num_classes=num_classes)

        # Used to calculate the class prediction for the training extra loss.
        predictions = tf.nn.top_k(logits, k=1)

        # Loss calculation. N depends on the accuracy of the previous batch. Bad accurate will generate extra loss.
        loss = n * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_batch)
        cross_entropy_mean = tf.reduce_mean(loss, name='cross_entropy')
        tf.summary.scalar(name='loss', tensor=cross_entropy_mean)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(cross_entropy_mean, global_step=global_step,
                                      var_list=tf.global_variables('classify'))

        # Savers
        saver_ft = tf.train.Saver(tf.global_variables('classify'))

        # Initializer
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            # Run the graph with value of n = 0
            sess.run(init)

            # Tensorborad options
            train_writer = tf.summary.FileWriter(FLAGS.log_dir, g)

            logger = init_logger()
            logger.info("Training starts...")

            # Training loop. Set the max number of steps.
            for epoch in range(0, FLAGS.max_steps):
                # We compute the image and label batch
                predicted, labels_batch1 = sess.run([predictions, labels_batch], feed_dict={n: 1.0})
                j = 1
                for i in range(batch_size):
                    if predicted.indices[i] == 1 and labels_batch1[i] == 0:
                        j += 1

                # Merge all summary variables for Tensorborad
                merge = tf.summary.merge_all()

                # We do the training and compute the loss and the summaries
                _, loss_val, summary = sess.run([train_op, cross_entropy_mean, merge], feed_dict={n: j})

                # Tensorboard update
                if epoch % 10 is 0:
                    logger.info('Time: %s   Loss: %f   Step: %i', str(datetime.now()), loss_val, epoch)
                    # Write the summaries in the log file
                    train_writer.add_summary(summary, epoch)

                # We save the progress every 500 steps
                if epoch % 500 is 0 and epoch is not 0:
                    saver_ft.save(sess, FLAGS.save_dir, global_step=global_step)
                    logger.info("***** Saving model in: %s *****", FLAGS.save_dir)

            logger.info("Training ends...")
            saver_ft.save(sess, FLAGS.save_dir, global_step=global_step)
            logger.info("***** Saving model in: %s *****", FLAGS.save_dir)


def main(argv=None):
    train()


if __name__ == "__main__":
    tf.app.run()

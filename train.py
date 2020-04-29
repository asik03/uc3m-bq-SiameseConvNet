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

from datetime import datetime

# # Hiperparameters for the training step
num_classes = 2  # Number of neurons in the final layer of the net.

# dropout_keep_prob = 0.85  # Estimated proportion of neurons to be kept from the dropout. Dropout equals 1 - dropout_keep_prob.
# learning_rate = 0.001  # Learning rate of the optimizer.
# batch_size = 16  # Number of elements of input on each "round".
# seed = 31  # Value used to set a random fixed value to the random variables.
# max_steps = 4000  # Number of epochs to run.


def init_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

# To open tensorboard server:   tensorboard --logdir=./data/mobilenetv3/logs/ --port 6006

def deploy(model_name=None, seed=None, batch_size=32, max_steps=2000, dropout=0.85, learning_rate=0.001):
    print("------- Training model -------")
    print("Model: " + model_name)
    print("Seed: " + str(seed))
    print("Batch_size: " + str(batch_size))
    print("Max_steps: " + str(max_steps))
    print("Dropout: " + str(dropout))
    print("Learning_rate: " + str(learning_rate))

    if model_name == "inceptionresnetv1":
        feature_lenght = 1792
        from model import inception_resnet_v1 as model
    elif model_name == "mobilenetv2":
        feature_lenght = 1280
        from model import mobilenetv2 as model
    elif model_name == "mobilenetv3":
        feature_lenght = 1280
        from model import mobilenetv3 as model
    else:
        raise ValueError("The model " + str(model_name) + " doesn't exist.")

    tfrecord_file = "./data/" + model_name + "/tfrecord_train_file"  # File with the dataset to train.
    # Directory where to write event logs. Structure: './data/modelName/logs/logs_seed_batchSize_maxSteps_dropout_learningRate'
    log_dir = "./data/" + model_name + "/logs/logs_" + str(seed) + "_" + str(batch_size) + "_" + str(max_steps) + "_" + \
              str(dropout) + "_" + str(learning_rate) + "/"
    # Directory where to save and load the checkpoints. Structure: './data/modelName/logs/logs_seed_batchSize_maxSteps_dropout_learningRate'
    save_dir = "./data/" + model_name + "/saves/saves_" + str(seed) + "_" + str(batch_size) + "_" + str(max_steps) + "_" + \
               str(dropout) + "_" + str(learning_rate) + "/"

    if not os.path.exists(save_dir):
        if not os.path.exists("./data/" + model_name + "/saves/"):
            os.makedirs("./data/" + model_name + "/saves/")
        os.mkdir(save_dir)
    with tf.Graph().as_default() as g:
        global_step = tf.train.get_or_create_global_step()
        n = tf.placeholder(tf.float32)
        tf.set_random_seed(seed)

        # Get the bottlenecks and labels from the dataset using the iterator
        iterator = data.create_iterator_for_diff(tfrecord_file, is_training=True, batch_size=batch_size,
                                                 f_lenght=feature_lenght)
        bottlenecks_1_batch, bottlenecks_2_batch, labels_batch = iterator.get_next()
        print(bottlenecks_1_batch)

        # Get the absolute difference bottlenecks, using tensorflow functions.
        diff_bottlenecks_tensor = tf.abs(tf.subtract(bottlenecks_1_batch, bottlenecks_2_batch))
        print(diff_bottlenecks_tensor)
        # Obtain the logits from the bottlenecks difference.
        logits = model.classify_bottlenecks(diff_bottlenecks_tensor, dropout_keep_prob=dropout,
                                            num_classes=num_classes, is_training=True)

        # Used to calculate the class prediction for the training extra loss.
        predictions = tf.nn.top_k(logits, k=1)

        # Loss calculation. N depends on the accuracy of the previous batch. Bad accurate will generate extra loss.
        loss = n * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_batch)
        cross_entropy_mean = tf.reduce_mean(loss, name="cross_entropy")
        tf.summary.scalar(name="loss", tensor=cross_entropy_mean)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(cross_entropy_mean, global_step=global_step,
                                      var_list=tf.global_variables("classify"))

        # Savers
        saver_ft = tf.train.Saver(tf.global_variables("classify"))

        # Initializer
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            # Run the graph with value of n = 0
            sess.run(init)

            # Tensorborad options
            train_writer = tf.summary.FileWriter(log_dir, g)

            logger = init_logger()
            logger.info("Training starts...")

            # Training loop. Set the max number of steps.
            for epoch in range(0, max_steps):
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
                    logger.info("Time: %s   Loss: %f   Step: %i", str(datetime.now()), loss_val, epoch)
                    # Write the summaries in the log file
                    train_writer.add_summary(summary, epoch)

                # We save the progress every 500 steps
                if epoch % 500 is 0 and epoch is not 0:
                    saver_ft.save(sess, save_dir, global_step=global_step)
                    logger.info("***** Saving model in: %s *****", save_dir)

            logger.info("Training ends...")
            saver_ft.save(sess, save_dir, global_step=global_step)
            logger.info("***** Saving model in: %s *****", save_dir)


def main(argv=None):
    deploy("mobilenetv3", 31, 16, 2000, 0.85, 0.001)


if __name__ == "__main__":
    tf.app.run()

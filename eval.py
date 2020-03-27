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


num_classes = 2  # Number of neurons in the final layer of the net.
root_path = "D:/PycharmProjects/uc3m-bq-SiameseConvNet/"
# dropout_keep_prob = 0.85  # Estimated proportion of neurons to be kept from the dropout. Dropout equals 1 - dropout_keep_prob.
# batch_size = 1  # Number of elements of input on each "round".
# success_constraint = 0.999  # Used to set the success boundary to consider same person in both images.
# seed = 31  # Value used to set a random fixed value to the random variables.
#
# train_batch_size = 16


def deploy(model_name=None, seed=None, batch_size=32, max_steps=2000, dropout=0.85, learning_rate=0.001,
           success_boundary=None):
    print("------- Training model -------")
    print("Model: " + model_name)
    print("Seed: " + str(seed))
    print("Batch_size: " + str(batch_size))
    print("Max_steps: " + str(max_steps))
    print("Dropout: " + str(dropout))
    print("Learning_rate: " + str(learning_rate))
    print("Success_boundary: " + str(success_boundary))

    if model_name == "inceptionresnetv1":
        feature_lenght = 1792
        from model import inception_resnet_v1 as model
    elif model_name == "mobilenetv2":
        feature_lenght = 1280
        from model import mobilenetv2 as model
    else:
        raise ValueError("The model " + model_name + " doesn't exist.")

    # Directory where to save and load the checkpoints. Structure: './data/modelName/logs/logs_seed_batchSize_maxSteps_dropout_learningRate'
    save_dir = root_path + "data/" + model_name + "/saves/saves_" + str(seed) + "_" + str(batch_size) + "_" + str(max_steps) + "_" + \
               str(dropout) + "_" + str(learning_rate) + "/"
    # File with the dataset to train. Bottlenecks in th is case
    tfrecord_file = root_path + 'data/' + model_name + '/tfrecord_test_file'

    with tf.Graph().as_default():
        tf.set_random_seed(seed)

        # Create dataset iterator of batch 1, obtaining the statistics correctly
        iterator = data.create_iterator_for_diff(tfrecord_file, is_training=False, batch_size=batch_size,
                                                 f_lenght=feature_lenght)
        bottlenecks_1_batch, bottlenecks_2_batch, labels_batch = iterator.get_next()

        # Get the bottleneck distances between each of them, and transform to positive their values.
        diff_bottlenecks_tensor = tf.abs(tf.subtract(bottlenecks_1_batch, bottlenecks_2_batch))

        # Get the logits from the classify model.
        logits = model.classify_bottlenecks(diff_bottlenecks_tensor, num_classes=num_classes, is_training=False)

        # Get the class with the highest score
        predictions = tf.nn.top_k(logits, k=1)

        # Variables initialisation
        init = tf.global_variables_initializer()

        # Create the saver to load the model weights and bias.
        saver = tf.train.Saver(tf.global_variables('classify'))

        logger = init_logger()
        logger.info("Eval starts...")

        with tf.Session() as sess:
            sess.run(init)

            # Restoring the classifier model
            saver.restore(sess, tf.train.latest_checkpoint(save_dir))

            # Auxiliary variables
            predicted_label = 0
            false_positives = 0
            false_negatives = 0
            true_success = 0
            false_success = 0
            exec_next_step = True

            while exec_next_step:
                try:
                    # Obtain the prediction(s) and the label(s) of each step.
                    predicted, label = sess.run([predictions, labels_batch])

                    # Using auxiliary parameter to check the results later.
                    if predicted.indices[0][0] == 0:
                        predicted_label = 0
                    elif predicted.indices[0][0] == 1:
                        if predicted.values[0][0] >= success_boundary:
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
                    logger.info("ACCURACY: %i ", (true_success + false_success) / (
                            false_positives + false_negatives + true_success + false_success) * 100)
                    # ROC matrix
                    logger.info(success_boundary)
                    exec_next_step = False
                ## todo: Guardar resultados


def main(argv=None):
    deploy("mobilenetv2", 31, 16, 2000, 0.85, 0.001, 0.75)


def init_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


if __name__ == "__main__":
    tf.app.run()

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

from model import inception_resnet_v1 as model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('ckpt_dir', './data/saves/', """Directory where to save and load the checkpoints. """)


img_1_path = "/home/uc3m1/PycharmProjects/siameseFaceNet/data/datasets/filtered_eval_dataset/Abel_Pacheco/Abel_Pacheco_0001.png"
img_2_path = "/home/uc3m1/PycharmProjects/siameseFaceNet/data/datasets/dataset_prueba/Amelie_Mauresmo/Amelie_Mauresmo_0001.png"

dropout_keep_prob = 0.8
num_classes = 2
image_size = 182


def predict(img_1, img_2):

    with tf.Graph().as_default():
        img_1 = tf.read_file(img_1)
        img_1 = tf.image.decode_png(img_1, channels=3)
        img_1 = tf.image.resize_images(img_1, (image_size, image_size))
        img_1 = tf.image.per_image_standardization(img_1)
        img_1 = tf.expand_dims(img_1, 0)

        img_2 = tf.read_file(img_2)
        img_2 = tf.image.decode_png(img_2, channels=3)
        img_2 = tf.image.resize_images(img_2, (image_size, image_size))
        img_2 = tf.image.per_image_standardization(img_2)
        img_2 = tf.expand_dims(img_2, 0)
        # TODO: Align face an crop it.

        bottleneck_1, end_points_1 = model.resnet_bottleneck(img_1, phase_train=False)
        bottleneck_2, end_points_2 = model.resnet_bottleneck(img_2, phase_train=False)

        diff_bottlenecks_tensor = tf.abs(tf.subtract(bottleneck_1, bottleneck_2))

        logits = model.classify_bottlenecks(diff_bottlenecks_tensor, dropout_keep_prob, num_classes=num_classes)

        # Get the class with the highest score
        predictions = tf.nn.top_k(logits, k=1)

        # Savers creations
        saver_bottleneck = tf.train.Saver(tf.global_variables('InceptionResnetV1'))
        saver_classify = tf.train.Saver(tf.global_variables('classify'))

        init = tf.global_variables_initializer()

        logger = init_logger()
        logger.info("Predict starts...")

        with tf.Session() as sess:
            sess.run(init)

            saver_bottleneck.restore(sess, './data/weights/model-20180408-102900.ckpt-90')

            saver_classify.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))

            predicted = sess.run(predictions)

            pred_label = predicted.indices[0][0]

            if pred_label == 1:
                logger.info("Both images are the same person")
            else:
                logger.info("Both images are not the same person")


def main(argv=None):
    predict(img_1_path, img_2_path)


def init_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


if __name__ == "__main__":
    tf.app.run()
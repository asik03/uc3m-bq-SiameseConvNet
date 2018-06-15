from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import logging
from model import inception_resnet_v1 as model
import numpy as np
import cv2
from scipy import misc

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('ckpt_dir', './data/weights/model-20180408-102900.ckpt-90', """Directory where to restore a model""")
tf.app.flags.DEFINE_string('log_dir', './prueba1/logs/', """Directory where to write event logs """)
tf.app.flags.DEFINE_integer('max_steps', 500, """Number of epochs to run.""")
tf.app.flags.DEFINE_string('save_dir', './prueba1/save/', """Directory where to load the fine tuning checkpoints """)


def init_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


def inference():
    with tf.Graph().as_default() as g:
        # Load an image into a numpy array, and expand the dimension to the correct one, based on the
        # inception_resNet model.
        img = cv2.imread(
            "/home/uc3m1/PycharmProjects/siameseFaceNet/testing_images/players_daschle_tom.jpg")
        img = cv2.resize(img, (182, 182))
        print(img.size)
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        # Making the bottlenecks
        bottleneck, end_points = model.inference(img)

        # Obtaining the logits of the fully connected layer with the bottlenecks
        logits = model.fine_tuning(bottleneck, end_points, num_classes=7)
        # TODO: automatizr el numero de clases en funcion de las que tengamosen cada caso

        print("Bottleneck: ", bottleneck)
        print("Logits: ", logits)

        # Get the class with the top 5 highest score
        predictions = tf.nn.top_k(logits, k=5)

        # Savers
        saver = tf.train.Saver(tf.global_variables("InceptionResnetV1"))
        # todo: automatizar nombre del modelo
        saver_ft = tf.train.Saver(tf.global_variables('fine_tuning'))

        # Initializer
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            print(sess.run(bottleneck))
            saver.restore(sess, FLAGS.ckpt_dir)
            saver_ft.restore(sess, tf.train.latest_checkpoint(FLAGS.save_dir))

            predicted = sess.run([predictions])
            print(predicted)


def main(argv=None):
    inference()


if __name__ == "__main__":
    tf.app.run()

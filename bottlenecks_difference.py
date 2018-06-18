from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import logging
from model import inception_resnet_v1 as model
import numpy as np
import cv2
import sys
import argparse

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('ckpt_dir', '/home/uc3m1/PycharmProjects/siameseFaceNet/data/weights/model-20180408-102900.ckpt-90', """Directory where to restore a model""")
tf.app.flags.DEFINE_string('log_dir', '/home/uc3m1/PycharmProjects/siameseFaceNet/prueba1/logs/', """Directory where to write event logs """)
tf.app.flags.DEFINE_integer('max_steps', 500, """Number of epochs to run.""")
tf.app.flags.DEFINE_string('save_dir', '/home/uc3m1/PycharmProjects/siameseFaceNet/prueba1/save/', """Directory where to load the fine tuning checkpoints """)


def init_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


def get_difference(img_1, img_2):
    with tf.Graph().as_default() as g:
        # TODO: ALINEAR CARA
        img1 = cv2.imread(img_1)
        img2 = cv2.imread(img_2)

        img1 = cv2.resize(img1, (182, 182))
        img2 = cv2.resize(img2, (182, 182))

        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)

        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        # Making the bottlenecks
        bottleneck1, end_points1 = model.inference(img1, reuse=True)
        bottleneck2, end_points2 = model.inference(img2, reuse=True)

        print("Bottleneck1: ", bottleneck1)
        print("Bottleneck2: ", bottleneck2)

        # Savers
        saver = tf.train.Saver(tf.global_variables("InceptionResnetV1"))
        # todo: automatizar nombre del modelo

        # Initializer
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            saver.restore(sess, FLAGS.ckpt_dir)

            bottleneck1 = sess.run(bottleneck1)
            bottleneck2 = sess.run(bottleneck2)
            diff = np.asarray(bottleneck1)-np.asarray(bottleneck2)
            return diff


def main(args):
    paths_1, paths_2, labels = load_paths_and_labels(args.first_dataset)
    print(paths_1, paths_2, labels)
    print(len(paths_1))
    print(len(paths_2))
    print(len(labels))


'''
    im1 = '/home/uc3m1/PycharmProjects/siameseFaceNet/datasets/dataset_prueba/Arnold_Schwarzenegger/Arnold_Schwarzenegger_0004.png'
    im2 = '/home/uc3m1/PycharmProjects/siameseFaceNet/datasets/dataset_prueba/Amelie_Mauresmo/Amelie_Mauresmo_0008.png'
    diff = get_difference(im1, im2)
    print(diff)
    print(diff.size)
'''


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--first_dataset', type=str, help='Path to the data text containing 3 size tuples')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

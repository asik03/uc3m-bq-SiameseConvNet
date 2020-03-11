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
import imutils
import dlib
import cv2

from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from model import inception_resnet_v1 as model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('ckpt_dir', './data/saves/', """Directory where to save and load the checkpoints. """)

# Path of the images to compare.
img_1_path = "/home/uc3m1/PycharmProjects/siameseFaceNet/data/test/con gafas.jpeg"
img_2_path = "/home/uc3m1/PycharmProjects/siameseFaceNet/data/test/sin gafas.jpeg"

# The path to dlib’s pre-trained facial landmark detector.
predictor_path = "/home/uc3m1/PycharmProjects/siameseFaceNet/data/align/shape_predictor_68_face_landmarks.dat"

# Model parameters.
num_classes = 2             # Number of neurons in the final layer of the net.
dropout_keep_prob = 0.85    # Estimated proportion of neurons to be kept from the dropout. Dropout equals 1 - dropout_keep_prob.
image_size = 182            # Used to set the face align before the prediction.
success_constraint = 0.99   # Used to set the success boundary to consider same person in both images.


def align_face(img_path, img_size):
    # Initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner.
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    fa = FaceAligner(predictor, desiredFaceWidth=img_size)

    # Load the input image, resize it, and convert it to grayscale.
    image = cv2.imread(img_path)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 2)

    # Loop over the face detections.
    for rect in rects:
        # Extract the ROI of the *original* face, then align the face using facial landmarks.
        (x, y, w, h) = rect_to_bb(rect)
        faceAligned = fa.align(image, gray, rect)

    return faceAligned


def predict(img_1, img_2, success_constraint):

    global predicted_label
    with tf.Graph().as_default():
        # Align faces, image standardization and dimension resize with a new header dimension.
        img_1 = align_face(img_1, image_size)
        img_1 = tf.image.per_image_standardization(img_1)
        img_1 = tf.expand_dims(img_1, 0)

        img_2 = align_face(img_2, image_size)
        img_2 = tf.image.per_image_standardization(img_2)
        img_2 = tf.expand_dims(img_2, 0)

        # Make both bottleneck inferences.
        bottleneck_1, end_points_1 = model.resnet_bottleneck(img_1, phase_train=False)
        bottleneck_2, end_points_2 = model.resnet_bottleneck(img_2, phase_train=False)

        # Absolute difference between both bottlenecks.
        diff_bottlenecks_tensor = tf.abs(tf.subtract(bottleneck_1, bottleneck_2))

        # Classify the diff bottleneck to obtain the final labels.
        logits = model.classify_bottlenecks(diff_bottlenecks_tensor, dropout_keep_prob, num_classes=num_classes, is_training=False)

        # Get the class with the highest score. Size of two values (pre and post softmax value normalization) and
        # indices (predicted label, 0 or 1)
        predictions = tf.nn.top_k(logits, k=1)

        # Savers creations.
        saver_bottleneck = tf.train.Saver(tf.global_variables('InceptionResnetV1'))
        saver_classify = tf.train.Saver(tf.global_variables('classify'))

        # Variables initialisation
        init = tf.global_variables_initializer()

        logger = init_logger()
        logger.info("Predict starts...")

        with tf.Session() as sess:
            sess.run(init)

            # Restore savers
            saver_bottleneck.restore(sess, './data/weights/model-20180408-102900.ckpt-90')
            saver_classify.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))

            # Running prediction tensor to obtain the results
            predicted = sess.run(predictions)

            # Using auxiliary parameter to check the results later.
            if predicted.indices[0][0] == 0:
                predicted_label = 0
            elif predicted.indices[0][0] == 1:
                if predicted.values[0][0] >= success_constraint:
                    predicted_label = 1
                else:
                    predicted_label = 0

            if predicted_label == 1:
                logger.info("Both images are the same person")
            else:
                logger.info("Both images are not the same person")
            print(predicted.values[0][0])


def main(argv=None):
    predict(img_1_path, img_2_path, success_constraint)


def init_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


if __name__ == "__main__":
    tf.app.run()
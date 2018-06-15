import tensorflow as tf

import numpy as np
import load_data as data
from model.inception_resnet_v1 import inference as resnet_bottleneck
from tensorflow.python.tools import inspect_checkpoint as chkp


def main():
    with tf.Graph().as_default():
        # TODO : ALINEAR IMAGEN
        iterator = data.create_iterator("./data/all_img_paths.txt")
        img, path = iterator.get_next()

        bottleneck_tensor, end_points = resnet_bottleneck(img)

        # Initializers
        init_global = tf.initializers.global_variables()
        init_local = tf.initializers.local_variables()

        # Checking if the checkpoints tensors
        chkp.print_tensors_in_checkpoint_file('/home/uc3m1/PycharmProjects/siameseFaceNet/data/weights/model-20180408'
                                              '-102900.ckpt-90', tensor_name='', all_tensors=False,
                                              all_tensor_names=True)

        # Create a saver
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

            sess.run(init_global)
            sess.run(init_local)

            # Restore the pretrained model from FaceNet
            saver.restore(sess, '/home/uc3m1/PycharmProjects/siameseFaceNet/data/weights/model-20180408-102900.ckpt-90')

            for i in range(0, 100000):
                bottleneck = sess.run(bottleneck_tensor)
                print("Bottleneck: ", bottleneck)


if __name__ == '__main__':
    main()

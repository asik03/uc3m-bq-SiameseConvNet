import tensorflow as tf
import numpy as np
import os
import load_data as data
from model.inception_resnet_v1 import inference as resnet_bottleneck


def main():
    with tf.Graph().as_default():

        iterator = data.create_iterator("./data/all_img_paths.txt")
        img, path_tensor = iterator.get_next()

        bottleneck_tensor, end_points = resnet_bottleneck(img, phase_train=False)

        dir_bottlenecks = "./data/bottlenecks/"

        if not os.path.exists(dir_bottlenecks):
            os.mkdir(dir_bottlenecks)

        # Initializers
        init_global = tf.initializers.global_variables()
        init_local = tf.initializers.local_variables()

        # chkp.print_tensors_in_checkpoint_file('./data/weights/model-20180408-102900.ckpt-90', tensor_name='',
        # all_tensors=False, all_tensor_names=True)

        # Create a saver
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

            sess.run(init_global)
            sess.run(init_local)

            # Restore the pretrained model from FaceNet
            saver.restore(sess, '/home/uc3m1/PycharmProjects/siameseFaceNet/data/weights/model-20180408-102900.ckpt-90')

            for i in range(0, 100000):
                try:
                    bottleneck, path = sess.run([bottleneck_tensor, path_tensor])
                    path = path[0].decode("utf-8")

                    basename = os.path.basename(path).strip('.png')
                    class_name = os.path.basename(os.path.dirname(path))
                    new_path = os.path.join(class_name, basename)
                    new_path = os.path.join(dir_bottlenecks, new_path)

                    if not os.path.exists(os.path.split(new_path)[0]):
                        os.mkdir(os.path.split(new_path)[0])

                    np.save(new_path, bottleneck)
                    print("Bottleneck", i, "saved at:", new_path)

                except tf.errors.OutOfRangeError:
                    print('Finished.')
                    break


if __name__ == '__main__':
    main()

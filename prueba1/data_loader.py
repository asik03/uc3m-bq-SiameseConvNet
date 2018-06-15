import tensorflow as tf
import numpy as np
import os
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('image_size', 182, """Height and width of the images.""")
tf.app.flags.DEFINE_string('data_path', "/home/uc3m1/PycharmProjects/siameseFaceNet/prueba1/test.txt", """Path to the data.""")
tfrecord_file_training = os.path.join(FLAGS.data_path, "flowers_train.tfrecord")
tfrecord_file_eval = os.path.join(FLAGS.data_path, "flowers_eval.tfrecord")


# Read a text file with a row in each line, and convert it into two numpy arrays with paths and labels.
def txt_to_np():
    with open(FLAGS.data_path, 'r') as f:
        lines = f.readlines()
        paths = []
        labels = []

        for i, line in enumerate(lines):
            path, label = line.split()
            paths.append(path)
            labels.append(label)



    return np.array(paths), np.array(labels)


# Read the image of a file path, and convert it into a Tensorflow input with some modifications.
def parse_data(path, label):
    file = tf.read_file(path)
    image = tf.image.decode_png(file, channels=3)
    image = tf.image.resize_images(image, (FLAGS.image_size, FLAGS.image_size))
    # TODO: Hacer data augmentation

    # Random crop image
    cropped_image = tf.image.resize_image_with_crop_or_pad(image, 324, 324)
    cropped_image = tf.random_crop(cropped_image, [FLAGS.image_size, FLAGS.image_size, 3])

    distorted_image = tf.image.random_brightness(cropped_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    norm_image = tf.image.per_image_standardization(distorted_image)
    new_label = tf.string_to_number(label)
    return tf.cast(norm_image, tf.float32), tf.cast(new_label, tf.int32)


def create_iterator(paths, labels):
    with tf.variable_scope('Iterator'):
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels)).repeat()
        dataset = dataset.map(parse_data)
        dataset = dataset.batch(32)

        iterator = dataset.make_one_shot_iterator()

    return iterator


def main():
    pass


if __name__ == '__main__':
    main()
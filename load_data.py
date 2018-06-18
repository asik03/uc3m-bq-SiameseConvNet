import numpy as np
import tensorflow as tf


def load_3_paths_and_labels(data_file):
    with open(data_file, 'r') as f:
        lines = f.readlines()
        paths1 = []
        paths2 = []
        labels = []

        for i, line in enumerate(lines):
            path1, path2, label = line.split()
            paths1.append(path1)
            paths2.append(path2)
            labels.append(label)

    return np.array(paths1), np.array(paths2), np.array(labels)


def _load_all_images_paths(data_file):
    with open(data_file, 'r') as f:
        lines = f.readlines()
        paths = []

        for i, line in enumerate(lines):
            paths.append(line.strip('\n'))
    return np.array(paths)


# Read the image of a file path, and convert it into a Tensorflow input with some modifications.
def _parse_data(path):
    file = tf.read_file(path)
    img = tf.image.decode_png(file, channels=3)
    img = tf.image.resize_images(img, (182, 182))

    # TODO : Preprocessing: ALINEAR IMAGEN. Suponemos que estan ya alineadas con la libreria de Facenet: /src/align/align_dataset_mtcnn.py

    img = tf.image.per_image_standardization(img)

    return img, path


def create_iterator(data):
    paths = _load_all_images_paths(data)

    with tf.variable_scope('Iterator'):
        dataset = tf.data.Dataset.from_tensor_slices(paths)
        dataset = dataset.map(_parse_data)
        dataset = dataset.batch(1)
        print("Dataset:", dataset)

        iterator = dataset.make_one_shot_iterator()
    return iterator


def main(argv= None):

    create_iterator("/home/uc3m1/PycharmProjects/siameseFaceNet/data/all_img_paths.txt")


if __name__ == '__main__':
    tf.app.run()


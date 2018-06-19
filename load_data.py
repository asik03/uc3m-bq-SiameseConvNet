import numpy as np
import tensorflow as tf


def _load_3_paths_and_labels(data_file):
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

        iterator = dataset.make_one_shot_iterator()
    return iterator


def _parse_data_for_diff(path1, path2, label):
    file_1 = tf.read_file(path1)
    file_2 = tf.read_file(path2)
    bottleneck_1 = np.load(file_1)
    bottleneck_2 = np.load(file_2)
    print(bottleneck_1)

    return np.bottleneck_1, bottleneck_2, label

'''
def create_iterator_for_diff(data):
    paths1, paths2, labels = _load_3_paths_and_labels(data)

    with tf.variable_scope('Iterator'):
        dataset = tf.data.Dataset.from_tensor_slices((tf.string_to_number(paths1), tf.string_to_number(paths2), tf.string_to_number(labels)))

        dataset_tensor = tf.convert_to_tensor(dataset, np.float32)

        dataset = dataset_tensor.map(_parse_data_for_diff)
        dataset = dataset.batch(32)
        print("Dataset:", dataset)

        iterator = dataset.make_one_shot_iterator()
    return iterator
'''


def create_iterator_for_diff(tfrecord_file, batch_size=64):
    """ Creates a one shot iterator from the TFRecord files.
        Args:
            tfrecord_file: a tensorflow record file path with the bottlenecks and labels.
            batch_size:
        Return:
            iterator: one shot iterator.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_file)

    dataset = dataset.map(parse)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=2560)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()

    return iterator


def parse(serialized):
    """ Convert the images and labels from records feature to Tensors.
        Args:
            serialized: A dataset comprising records from one TFRecord file.
    """
    # Define a dict with the data-names and types we expect to find in the TFRecords file.

    feature = {
        'bottleneck_1': tf.FixedLenFeature((1792,), tf.float32),
        'bottleneck_2': tf.FixedLenFeature((1792,), tf.float32),
        'label': tf.FixedLenFeature([], tf.int64),
    }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=feature)

    # Get the image as raw bytes, and  the height, width and label as int.
    bottleneck_1 = tf.cast(parsed_example['bottleneck_1'], tf.float32)
    bottleneck_2 = tf.cast(parsed_example['bottleneck_2'], tf.float32)
    label = tf.cast(parsed_example['label'], tf.int64)

    # The image and label are now correct TensorFlow types.
    return bottleneck_1, bottleneck_2, label


def main(argv= None):
    pass
    #create_iterator("/home/uc3m1/PycharmProjects/siameseFaceNet/data/all_img_paths.txt")
    #consume_tfrecord()


if __name__ == '__main__':
    tf.app.run()


import os
import tensorflow as tf
import numpy as np


data_dir = '/home/uc3m1/PycharmProjects/siameseFaceNet/data/datasets/filtered_dataset'
diff_dataset = "./data/diff_dataset.txt"
img_paths = "./data/all_img_paths.txt"


def generate_txt_with_all_images(data_dir, path):
    with open(path, 'w') as out:
        for person in os.listdir(data_dir):
            path = os.path.join(data_dir, person)

            for img in os.listdir(path):
                img_path = os.path.join(path, img)
                out.write(img_path + '\n')
                print(img_path)


def generate_tfrecord_files(dataset, save_file):
    """ Creates the tfrecord files from a dataset file.

        Args:
            dataset: txt file with lines having 'path_to_the_image label'.
            save_file: file where the TFRecord is going to be saved.
    """
    if os.path.exists(save_file):
        print("TFRecord file already exists in", save_file)
        return

    print("Creating TFRecord file...")

    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(save_file) as writer:

        for entry in open(dataset):
            tf_example = _create_tf_example(entry)
            writer.write(tf_example.SerializeToString())

    print("TFRecord file created at", save_file)


def _create_tf_example(entry):
    """ Creates a tf.train.Example to be saved in the TFRecord file.

        Args:
            entry: string containing the path to a image and its label.
        Return:
            tf_example: tf.train.Example containing the info stored in feature
    """
    image_path_1, image_path_2, label = _get_image_and_label_from_entry(entry)

    bottleneck_1 = np.load(image_path_1)
    bottleneck_2 = np.load(image_path_2)

    # Data which is going to be stored in the TFRecord file
    feature = {
        'bottleneck_1': tf.train.Feature(float_list=tf.train.FloatList(value=bottleneck_1.reshape(-1))),
        'bottleneck_2': tf.train.Feature(float_list=tf.train.FloatList(value=bottleneck_2.reshape(-1))),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

    return tf_example


def _get_image_and_label_from_entry(entry):
    """ Get the image's path and its label from a dataset entry.

        Args:
            entry: string containing the path to a image and its label.
        Return:
            file_path: string with the path where a image is stored.
            label: int representing the class of the image
    """
    file_path_1, file_path_2, label = entry.split(" ")

    label = label.strip('\n')

    return file_path_1, file_path_2, int(label)


def main():
    #generate_txt_with_all_images(data_dir, img_paths)

    generate_tfrecord_files(diff_dataset, "./data/tfrecord_train_file")
    generate_tfrecord_files(diff_dataset, "./data/tfrecord_eval_file")


if __name__ == '__main__':
    main()




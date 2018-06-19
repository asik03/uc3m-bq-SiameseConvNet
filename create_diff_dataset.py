import numpy as np
import load_data as data
import os
import random

data_file = "./data/test.txt"

num_tuples_per_class = 30


def main():

    basename_dir = "./data/bottlenecks/"
    with open('./data/diff_dataset.txt', 'w') as out:
        for class_name in os.listdir(basename_dir):
            class_path = os.path.join(basename_dir, class_name)
            print(os.listdir(class_path))

            for i in range(num_tuples_per_class):
                npy_path_1 = get_random_item_from_class(class_path)
                npy_path_2 = get_random_item_from_class(class_path)

                out.write(npy_path_1 + ' ' + npy_path_2 + ' 1' + '\n')
                print(npy_path_1, npy_path_2, '1')

            for i in range(num_tuples_per_class):
                npy_path_1 = get_random_item_from_class(class_path)

                comparing_class = get_random_class(basename_dir, exclude=os.listdir(basename_dir).index(class_name))
                npy_path_2 = get_random_item_from_class(os.path.join(basename_dir, comparing_class))

                out.write(npy_path_1 + ' ' + npy_path_2 + ' 0' + '\n')
                print(npy_path_1, npy_path_2, '0')


def get_random_item_from_class(class_path):
    rand = int(random.random() * len(os.listdir(class_path)))
    npy_path = os.listdir(class_path)[rand]

    return os.path.join(class_path, npy_path)


def get_random_class(basename_dir, exclude=None):
    rand = int(random.random() * len(os.listdir(basename_dir)))

    while rand == exclude:
        rand = int(random.random() * len(os.listdir(basename_dir)))

    class_path = os.listdir(basename_dir)[rand]

    return class_path


if __name__ == '__main__':
    main()

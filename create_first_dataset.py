
"""
This file will make a first siamese neural network dataset, based on 3 element for each row:
    - Image_path_1
    - Image_path_2
    - Label: if the image pairs are the same person, this label will be 1 (true), if both images are not the same
     person,this label will be 0 (false).

The dataset will be written in a text file, using a previous data structure like:
    - Name_1: a person directory with his images inside it
        - Img_1
        - Img_2
        - Img_3
    - Name_2
        - Img_1
        - Img_2
        - ...
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import sys
import numpy as np
import argparse
import random

# Prevent dependency problems
from random import shuffle
from facenet import facenet


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


# This method will return a list of the image directories, with the whole image path
def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = []
    # Filling in the "classes" list with de directory name of each class
    for path in os.listdir(path_exp):
        if os.path.isdir(os.path.join(path_exp, path)):
            classes.append(path)
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        # Getting the image paths of each image in each class
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = []
        for img in images:
            image_paths.append(os.path.join(facedir, img))
    return image_paths


def pop_random(lst):
    idx = random.randrange(0, len(lst))
    return lst.pop(idx)


def sample_people(dataset, people_per_batch, images_per_person, paths):
    nrof_images = people_per_batch * images_per_person

    # Sample classes from the dataset
    nrof_classes = len(dataset)
    print(nrof_classes)

    # Making  a numpy array from 0 to number of classes
    class_indices = np.arange(nrof_classes)
    print(class_indices)
    np.random.shuffle(class_indices)

    image_paths = []
    num_per_class = []
    sampled_class_indices = []

    # Sample images from these classes until we have enough
    # For each class
    for i in range(len(class_indices)):
        print(i)
        print(len(paths))
        print(len(image_paths))
        print(class_indices)
        print(" ")
        class_index = class_indices[i]
        # Number of images in each class
        nrof_images_in_class = len(dataset[class_index])
        # Array with size of n. n = number of images per class
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images - len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = []
        for j in idx:
            image_paths_for_class.append(dataset[class_index].image_paths[j])
        sampled_class_indices += [class_index] * nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i += 1
    return image_paths, num_per_class, class_indices


# Save the dataset into a text file
def save(n_dataset):
    file = open('./test.txt', 'w')
    for item in n_dataset:
        file.write("%s\n" % item)
    file.close()


def main(args):
    # Getting the dataset with facenet
    dataset = facenet.get_dataset(args.data_dir)

    paths, labels = facenet.get_image_paths_and_labels(dataset)

    # Sample people randomly from the dataset
    image_paths, num_per_class, class_indices = sample_people(dataset, args.people_per_batch, args.images_per_person,
                                                              paths)

    print('Starting...')
    start_time = time.time()

    new_dataset = []
    images_aux = image_paths

    # Random shuffle of the image paths
    shuffle(images_aux)

    # Creating the new dataset. First we pop out randomly an element from de paths array, and we compare it with the
    # rest of them. 1 if its the same directory(class), 0 if its the opposite.
    if os.path.exists("./data/test.txt"):
        os.remove("./data/test.txt")

    with open('./data/test.txt', 'w') as out:
        tuples = args.num_tuples_per_image
        while images_aux:
            count_true = 0
            count_false = 0
            rand1 = pop_random(images_aux)
            print(len(images_aux))
            for i in range(len(images_aux)):
                if os.path.dirname(rand1) == os.path.dirname(images_aux[i]):
                    if count_true == tuples:
                        continue
                    #new_dataset.append([rand1, images_aux[i], 1])
                    out.write(rand1 + ' ' + images_aux[i] + ' 1' + '\n')
                    count_true += 1

                else:
                    if count_false == tuples:
                        continue
                    #new_dataset.append([rand1, images_aux[i], 0])
                    out.write(rand1 + ' ' + images_aux[i] + ' 0' + '\n')
                    count_false += 1
                print(count_false, count_true)
            # When there is not more positive tuples than negatives, then it will stop
            if count_true != count_false:
                break

    end_time = time.time()
    total_time = end_time - start_time
    print("Finished. Total time: ", total_time)
    #print(new_dataset)
    #print(len(new_dataset))
    #save(new_dataset)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.', default='~/models/facenet')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--people_per_batch', type=int,
                        help='Number of people per batch.', default=45)
    parser.add_argument('--images_per_person', type=int,
                        help='Number of images per person.', default=40)
    parser.add_argument('--num_tuples_per_image', type=int,
                        help='Number of new dataset tuples per each image. For instance, a 4 value will add 4 tuples with positive label and 4 with negative label',
                        default=10)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

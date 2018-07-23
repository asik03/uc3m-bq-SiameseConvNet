# Convolutional Siamese Neural Network using Tensorflow

This is a [Tensorflow](https://www.tensorflow.org/) implementation based on [FaceNet](https://github.com/davidsandberg) and [ft_flowers](https://github.com/bq/uc3m/tree/master/ft_flowers) repositories made by David Sandberg and Juan Abascal and Daniel Gonzalez.

## Compatibility
The code was tested on Ubuntu 16.04 LTS, using Tensorflow r1.7 and Python 3.6.

## Dependences

## News
| Date  | Update |
| ------------- | ------------- |
| 2018-06-21  | First implementation  |

## How it's made

This model has been designed to operate in devices which not require any training step on them. The model is trained before it is used in that kind of hardware. s implementation can be 

This implementation has been done by using a Siamese Neural Network, which consist of a double inference of two images on a same neural network with both equal weight and bias. After that, the outputs are compared between them and it can be obtained a conclusion from the output distances. This is the traditional procedure on how can we implement a Siamese Neural Network, but this project has tested a new way to use this kind of architectures.

To understand it clearly, lets introduce the implementation step by step. The model is basically divided in two blocks: the convolutional and the fully-connected block:

IMAGE

As you can see on the image above, the image inputs are pre-processed with Face alignment using [Multi-task CNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html). On the training and evaluation procedures, the dataset to use are pre-processed completely with this implementation; and on the inference or predict procedure, it will be used on each inference. At the end of this step, the image is face-aligned and cropped in a N x N size.

After that, both pre-processed input images are incorporated to the convolutional phase. On this repo we used a modified [Inception-ResNet-V1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py)architecture, with their corresponded layers, excepting the final fully connected layer, which it's been removed from it. With this, what we obtain is an 1D array with 1792 positions that we can use for the next phase. This output is called _bottleneck_, and it can determine the features of the image that we will use to determine if both images are or not the same person.

## How to use it

### Filter dataset

On this step we will use the `filter_dataset.py` python script to filter the LFW dataset. We just need to change the parameters depending on what we want to consider for training.
* `create_filter_paths_txt(dataset_path, txt_with_new_dataset_train_paths, txt_with_new_dataset_eval_paths, min_num_images_per_class, max_num_images_per_class)`  
* `create_filtered_dataset(txt_with_new_dataset_train_paths, txt_with_new_dataset_eval_paths,filtered_dataset_train_path, filtered_dataset_eval_path)`


Anyways, the code that is needed to make it are commented with "#" in the main method. Just need to uncomment those lines we want to use and run the python file.

### Getting the bottlenecks
This step will make the bottleneck inferences from the filtered dataset we have just made on the previous step.

## Evaluation and results

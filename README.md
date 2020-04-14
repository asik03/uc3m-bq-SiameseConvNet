# Convolutional Siamese Neural Network using Tensorflow

This is a [Tensorflow](https://www.tensorflow.org/) implementation based on [FaceNet](https://github.com/davidsandberg) and [ft_flowers](https://github.com/bq/uc3m/tree/master/ft_flowers) repositories, made by David Sandberg and Juan Abascal and Daniel Gonzalez.

## Compatibility
The code was tested on Ubuntu 16.04 LTS and Windows 10, using Tensorflow r1.13.1 and Python 3.6.

## Requirements
Anaconda is suggested to make a virtual and independent python environment.
* Anaconda
* Tensorflow
* Numpy
* Cv2
* Dlib

## News
| Date  | Update |
| ------------- | ------------- |
| 2018-06-21  | First implementation  |
| 2020-04-14| Dynamic implementation of CNN block. Updated readme |
## How it's made

This model has been designed to operate in devices which do not require any training step on them. The model is trained before it is used in that kind of hardware.

This implementation has been done by using a Siamese Neural Network, which consist of a double inference of two images on a same neural network with both equal weight and bias. After that, the outputs are compared between them and it can be obtained a conclusion from the output distances. This is the traditional procedure on how can we implement a Siamese Neural Network, but this project has tested a new way to use this kind of architectures.

To understand it clearly, lets introduce the implementation step by step. The model is basically divided in two blocks: the convolutional and the fully-connected block:

![CSNN](https://i.imgur.com/viBXv55.jpg)

As you can see on the image above, firstly the image inputs are pre-processed with Face alignment using [Multi-task CNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html). On the training and evaluation procedures, the dataset to use are pre-processed completely with this implementation; on the inference or predict procedure, it will be used every time we do it. At the end of this step, the image is face-aligned and cropped in a N x N size.

After that, both pre-processed input images are incorporated to the convolutional phase. On this repo we used a modified [Inception-ResNet-V1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py)architecture, with their corresponded layers, excepting the final fully connected layer, which it's been removed from it. With this, what we obtain is an 1D array with 1792 positions that we can use for the next phase. However, in next implementations, a new dynamic feature allows to configure our own architecture. MobileNetV2 and MobileNetV3 are already implemented. This output is called _bottleneck_, and it can determine the features of the image that we will use to determine if both images are or not the same person.

Once the two bottlenecks are obtained, they are used for a new classification submodel. We make an euclidean distance between both arrays, obtaining a new positive values array called _Diff_. Then, the new classification consists of a fully connected model with a hidden layer of 512 neurons (this is not completely permanent, it can be modified for future improvements). The output consists of two classes, which indicates how strong is the similarity between the images.

## How to use it
### Prepairing data
#### Filter dataset

On this step we will use the `filter_dataset.py` python script to filter the LFW dataset. We just need to change the parameters depending on what we want to consider for training.
* `create_filter_paths_txt(dataset_path, txt_with_new_dataset_train_paths, txt_with_new_dataset_eval_paths, min_num_images_per_class, max_num_images_per_class)`  
* `create_filtered_dataset(txt_with_new_dataset_train_paths, txt_with_new_dataset_eval_paths,filtered_dataset_train_path, filtered_dataset_eval_path)`


Anyways, the code that is needed to make it are commented with "#" in the main method. Just need to uncomment those lines we want to use and run the python file.

#### Getting the bottlenecks
This step will make the bottleneck inferences from the filtered dataset we have just made on the previous step. These bottlenecks are obtained with pre-trained Inception-ResNet-V1 weights from faceNet [here](https://drive.google.com/file/d/1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz/view). We will make it for the training and evaluation dataset. We just need to open the `inference_bottlenecks.py` file and put or uncomment this commands:
* `inference_bottlenecks(img_paths_txt_train_path, bottlenecks_train_dir)`
* `inference_bottlenecks(img_paths_txt_eval_path, bottlenecks_eval_dir)`

After that, a two new directories will be created in the data path (this path can be modified alternatively).

#### Creating the classification tuples
Once the bottlenecks are generated, we use them to generate a text file with tuples of bottleneck pairs and a label on each line. The param `num_tuples_per_class` indicates the number of tuples it will be generated for each class. This can be run on the `pre_input` python file:
* ` create_diff_dataset_txt(bottlenecks_train_dir, diff_dataset_txt_train_path, num_tuples_per_class)`
* ` create_diff_dataset_txt(bottlenecks_eval_dir, diff_dataset_txt_eval_path, num_tuples_per_class=1)`

#### Generating TF-Record files
Creates the evaluation and training TF-Record dataset files used by the model, based on the bottlenecks generated before. On the `pre_input` file, add or uncomment the following code on the main method:
* `generate_tfrecord_files(diff_dataset_txt_train_path, "./data/tfrecord_train_file")`
* `generate_tfrecord_files(diff_dataset_txt_eval_path, "./data/tfrecord_eval_file")`

### Training the model
To train the model, we just need to load the training TF-Record dataset into the model, and run the `train.py` file. It's based on a fully connected classifier as we mention before. Inside `train.py`, we can modify the hiper-parameters: `max_steps`, `num_classes`, `dropout_keep_prob`, `learning_rate` and `batch_size`.

### Evaluating the model
Once the model is trained, we proceed to evaluate the results. From the `eval.py` script we can obtain different kind of results such us the confusion matrix. To obtain then we just need to run this python file.

## Evaluation and results



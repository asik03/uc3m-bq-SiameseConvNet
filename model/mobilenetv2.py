# Architecture based on MobileNetV2 https://arxiv.org/pdf/1801.04381.pdf

import tensorflow as tf
import tensorflow.contrib.slim as slim


def mobilenet_v2_arg_scope(weight_decay, is_training=True, depth_multiplier=1.0, regularize_depthwise=False,
                           dropout_keep_prob=1.0):
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None

    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'center': True, 'scale': True}):

        with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
            with slim.arg_scope([slim.separable_conv2d],
                                weights_regularizer=depthwise_regularizer, depth_multiplier=depth_multiplier):
                with slim.arg_scope([slim.dropout], is_training=is_training, keep_prob=dropout_keep_prob) as sc:
                    return sc


def block(net, input_filters, output_filters, expansion, stride):
    res_block = net
    res_block = slim.conv2d(inputs=res_block, num_outputs=input_filters * expansion, kernel_size=[1, 1])
    res_block = slim.separable_conv2d(inputs=res_block, num_outputs=None, kernel_size=[3, 3], stride=stride)
    res_block = slim.conv2d(inputs=res_block, num_outputs=output_filters, kernel_size=[1, 1], activation_fn=None)
    if stride == 2:
        return res_block
    else:
        if input_filters != output_filters:
            net = slim.conv2d(inputs=net, num_outputs=output_filters, kernel_size=[1, 1], activation_fn=None)
        return tf.add(res_block, net)


def blocks(net, expansion, output_filters, repeat, stride):
    input_filters = net.shape[3].value

    # first layer should take stride into account
    net = block(net, input_filters, output_filters, expansion, stride)

    for _ in range(1, repeat):
        net = block(net, input_filters, output_filters, expansion, 1)

    return net


def compute_bottleneck(images, keep_probability=0.8, phase_train=True, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=slim.initializers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return mobilenet_v2(images, is_training=phase_train, dropout_keep_prob=keep_probability)


def mobilenet_v2(inputs,
                 num_classes=10, # Fixed to 10 at the moment, needed for the pretrained weights used
                 dropout_keep_prob=0.999,
                 is_training=True,
                 depth_multiplier=1.0,
                 prediction_fn=tf.contrib.layers.softmax,
                 spatial_squeeze=True,
                 scope='MobilenetV2'):
    endpoints = dict()

    expansion = 6

    with tf.variable_scope(scope):

        with slim.arg_scope(mobilenet_v2_arg_scope(0.0004, is_training=is_training, depth_multiplier=depth_multiplier,
                                                   dropout_keep_prob=dropout_keep_prob)):
            net = tf.identity(inputs)

            net = slim.conv2d(net, 32, [3, 3], scope='conv11', stride=2)

            net = blocks(net=net, expansion=1, output_filters=16, repeat=1, stride=1)

            net = blocks(net=net, expansion=expansion, output_filters=24, repeat=2, stride=2)

            net = blocks(net=net, expansion=expansion, output_filters=32, repeat=3, stride=2)

            net = blocks(net=net, expansion=expansion, output_filters=64, repeat=4, stride=2)

            net = blocks(net=net, expansion=expansion, output_filters=96, repeat=3, stride=1)

            net = blocks(net=net, expansion=expansion, output_filters=160, repeat=3, stride=2)

            net = blocks(net=net, expansion=expansion, output_filters=320, repeat=1, stride=1)

            net = slim.conv2d(net, 1280, [1, 1], scope='last_bottleneck')

            bottleneck = slim.avg_pool2d(net, [7, 7])

            print(bottleneck)
            print(tf.shape(bottleneck))

            features = slim.conv2d(bottleneck, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='features')

            print(features)
            print(tf.shape(features))

            # net = slim.avg_pool2d(features, features.get_shape()[1:3], padding='VALID')

            bottleneck = slim.flatten(bottleneck)
            bottleneck = slim.dropout(bottleneck, dropout_keep_prob, is_training=is_training, scope='Dropout')
            print(bottleneck)
            print(tf.shape(bottleneck))
            # elimina las dimensiones de tamaño 1
#            if spatial_squeeze:
#                net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

            endpoints['Logits'] = features
            endpoints['PreLogitsFlatten'] = bottleneck

            #softmax
            if prediction_fn:
                endpoints['Predictions'] = prediction_fn(net, scope='Predictions')

    return bottleneck, endpoints


def classify_bottlenecks(diff_bottlenecks_tensor, dropout_keep_prob=0.85, num_classes=2, is_training=True):
    """
    Creates the classifier model.
        Args:
          diff_bottlenecks_tensor: a 1-D tensor of size [num_values_per_bottleneck].
          dropout_keep_prob: float, the fraction to keep before final layer.
          num_classes: number of predicted classes.
          is_training: whether is training or not.
        Returns:
          net: the logits outputs of the model.
          end_points[pre_softmax]: previous original values after the softmax step.
    """
    with tf.variable_scope('classify'):
        end_points = {}

        net = slim.flatten(diff_bottlenecks_tensor, scope='Flatten_1')

        end_points['Flatten_1'] = net

        # Creates a fully connected layer
        net = slim.fully_connected(net, 512, activation_fn=tf.nn.sigmoid,
                                   weights_initializer=tf.truncated_normal_initializer(stddev=0.01, seed=31),
                                   scope='FC_1')

        tf.summary.histogram(name='Weights_1',
                             values=tf.get_default_graph().get_tensor_by_name('classify/FC_1/weights:0'))
        tf.summary.histogram(name='Biases_1',
                             values=tf.get_default_graph().get_tensor_by_name('classify/FC_1/biases:0'))

        net = slim.dropout(net, dropout_keep_prob, scope='Dropout', is_training=is_training)

        pre_softmax = slim.fully_connected(net, num_classes, activation_fn=None,
                                   weights_initializer=tf.truncated_normal_initializer(stddev=0.01, seed=31),
                                   scope='FC_2')

        tf.summary.histogram(name='Weights_2',
                             values=tf.get_default_graph().get_tensor_by_name('classify/FC_2/weights:0'))
        tf.summary.histogram(name='Biases_2',
                             values=tf.get_default_graph().get_tensor_by_name('classify/FC_2/biases:0'))

        end_points['pre_softmax'] = pre_softmax

        net = slim.softmax(pre_softmax)

        end_points['Logits'] = net

        tf.add_to_collection('classify', net)

    return net


mobilenet_v2.default_image_size = 224

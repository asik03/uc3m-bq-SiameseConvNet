# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Mobilenet V3 conv defs and helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools
import numpy as np

import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

from model.mobnetv3utils import conv_blocks as ops
from model.mobnetv3utils import mobilenet as lib

slim = contrib_slim
op = lib.op
expand_input = ops.expand_input_by_factor

# Squeeze Excite with all parameters filled-in, we use hard-sigmoid
# for gating function and relu for inner activation function.
squeeze_excite = functools.partial(
    ops.squeeze_excite, squeeze_factor=4,
    inner_activation_fn=tf.nn.relu,
    gating_fn=lambda x: tf.nn.relu6(x + 3) * 0.16667)

# Wrap squeeze excite op as expansion_transform that takes
# both expansion and input tensor.
_se4 = lambda expansion_tensor, input_tensor: squeeze_excite(expansion_tensor)


def hard_swish(x):
    with tf.compat.v1.name_scope('hard_swish'):
        return x * tf.nn.relu6(x + np.float32(3)) * np.float32(1. / 6.)


def reduce_to_1x1(input_tensor, default_size=7, **kwargs):
    h, w = input_tensor.shape.as_list()[1:3]
    if h is not None and w == h:
        k = [h, h]
    else:
        k = [default_size, default_size]
    return slim.avg_pool2d(input_tensor, kernel_size=k, **kwargs)


def mbv3_op(ef, n, k, s=1, act=tf.nn.relu, se=None, **kwargs):
    """Defines a single Mobilenet V3 convolution block.

  Args:
    ef: expansion factor
    n: number of output channels
    k: stride of depthwise
    s: stride
    act: activation function in inner layers
    se: squeeze excite function.
    **kwargs: passed to expanded_conv

  Returns:
    An object (lib._Op) for inserting in conv_def, representing this operation.
  """
    return op(
        ops.expanded_conv,
        expansion_size=expand_input(ef),
        kernel_size=(k, k),
        stride=s,
        num_outputs=n,
        inner_activation_fn=act,
        expansion_transform=se,
        **kwargs)


def mbv3_fused(ef, n, k, s=1, **kwargs):
    """Defines a single Mobilenet V3 convolution block.

  Args:
    ef: expansion factor
    n: number of output channels
    k: stride of depthwise
    s: stride
    **kwargs: will be passed to mbv3_op

  Returns:
    An object (lib._Op) for inserting in conv_def, representing this operation.
  """
    expansion_fn = functools.partial(slim.conv2d, kernel_size=k, stride=s)
    return mbv3_op(
        ef,
        n,
        k=1,
        s=s,
        depthwise_location=None,
        expansion_fn=expansion_fn,
        **kwargs)


mbv3_op_se = functools.partial(mbv3_op, se=_se4)

DEFAULTS = {
    (ops.expanded_conv,):
        dict(
            normalizer_fn=slim.batch_norm,
            residual=True),
    (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
        'normalizer_fn': slim.batch_norm,
        'activation_fn': tf.nn.relu,
    },
    (slim.batch_norm,): {
        'center': True,
        'scale': True
    },
}

# Compatible checkpoint: http://mldash/5511169891790690458#scalars
V3_LARGE = dict(
    defaults=dict(DEFAULTS),
    spec=([
        # stage 1
        op(slim.conv2d, stride=2, num_outputs=16, kernel_size=(3, 3),
           activation_fn=hard_swish),
        mbv3_op(ef=1, n=16, k=3),
        mbv3_op(ef=4, n=24, k=3, s=2),
        mbv3_op(ef=3, n=24, k=3, s=1),
        mbv3_op_se(ef=3, n=40, k=5, s=2),
        mbv3_op_se(ef=3, n=40, k=5, s=1),
        mbv3_op_se(ef=3, n=40, k=5, s=1),
        mbv3_op(ef=6, n=80, k=3, s=2, act=hard_swish),
        mbv3_op(ef=2.5, n=80, k=3, s=1, act=hard_swish),
        mbv3_op(ef=184 / 80., n=80, k=3, s=1, act=hard_swish),
        mbv3_op(ef=184 / 80., n=80, k=3, s=1, act=hard_swish),
        mbv3_op_se(ef=6, n=112, k=3, s=1, act=hard_swish),
        mbv3_op_se(ef=6, n=112, k=3, s=1, act=hard_swish),
        mbv3_op_se(ef=6, n=160, k=5, s=2, act=hard_swish),
        mbv3_op_se(ef=6, n=160, k=5, s=1, act=hard_swish),
        mbv3_op_se(ef=6, n=160, k=5, s=1, act=hard_swish),
        op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=960,
           activation_fn=hard_swish),
        op(reduce_to_1x1, default_size=7, stride=1, padding='VALID'),
        op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280,
           normalizer_fn=None, activation_fn=hard_swish)
    ]))

# 72.2% accuracy.
V3_LARGE_MINIMALISTIC = dict(
    defaults=dict(DEFAULTS),
    spec=([
        # stage 1
        op(slim.conv2d, stride=2, num_outputs=16, kernel_size=(3, 3)),
        mbv3_op(ef=1, n=16, k=3),
        mbv3_op(ef=4, n=24, k=3, s=2),
        mbv3_op(ef=3, n=24, k=3, s=1),
        mbv3_op(ef=3, n=40, k=3, s=2),
        mbv3_op(ef=3, n=40, k=3, s=1),
        mbv3_op(ef=3, n=40, k=3, s=1),
        mbv3_op(ef=6, n=80, k=3, s=2),
        mbv3_op(ef=2.5, n=80, k=3, s=1),
        mbv3_op(ef=184 / 80., n=80, k=3, s=1),
        mbv3_op(ef=184 / 80., n=80, k=3, s=1),
        mbv3_op(ef=6, n=112, k=3, s=1),
        mbv3_op(ef=6, n=112, k=3, s=1),
        mbv3_op(ef=6, n=160, k=3, s=2),
        mbv3_op(ef=6, n=160, k=3, s=1),
        mbv3_op(ef=6, n=160, k=3, s=1),
        op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=960),
        op(reduce_to_1x1, default_size=7, stride=1, padding='VALID'),
        op(slim.conv2d,
           stride=1,
           kernel_size=[1, 1],
           num_outputs=1280,
           normalizer_fn=None)
    ]))

# Compatible run: http://mldash/2023283040014348118#scalars
V3_SMALL = dict(
    defaults=dict(DEFAULTS),
    spec=([
        # stage 1
        op(slim.conv2d, stride=2, num_outputs=16, kernel_size=(3, 3),
           activation_fn=hard_swish),
        mbv3_op_se(ef=1, n=16, k=3, s=2),
        mbv3_op(ef=72. / 16, n=24, k=3, s=2),
        mbv3_op(ef=(88. / 24), n=24, k=3, s=1),
        mbv3_op_se(ef=4, n=40, k=5, s=2, act=hard_swish),
        mbv3_op_se(ef=6, n=40, k=5, s=1, act=hard_swish),
        mbv3_op_se(ef=6, n=40, k=5, s=1, act=hard_swish),
        mbv3_op_se(ef=3, n=48, k=5, s=1, act=hard_swish),
        mbv3_op_se(ef=3, n=48, k=5, s=1, act=hard_swish),
        mbv3_op_se(ef=6, n=96, k=5, s=2, act=hard_swish),
        mbv3_op_se(ef=6, n=96, k=5, s=1, act=hard_swish),
        mbv3_op_se(ef=6, n=96, k=5, s=1, act=hard_swish),
        op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=576,
           activation_fn=hard_swish),
        op(reduce_to_1x1, default_size=7, stride=1, padding='VALID'),
        op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1024,
           normalizer_fn=None, activation_fn=hard_swish)
    ]))

# 62% accuracy.
V3_SMALL_MINIMALISTIC = dict(
    defaults=dict(DEFAULTS),
    spec=([
        # stage 1
        op(slim.conv2d, stride=2, num_outputs=16, kernel_size=(3, 3)),
        mbv3_op(ef=1, n=16, k=3, s=2),
        mbv3_op(ef=72. / 16, n=24, k=3, s=2),
        mbv3_op(ef=(88. / 24), n=24, k=3, s=1),
        mbv3_op(ef=4, n=40, k=3, s=2),
        mbv3_op(ef=6, n=40, k=3, s=1),
        mbv3_op(ef=6, n=40, k=3, s=1),
        mbv3_op(ef=3, n=48, k=3, s=1),
        mbv3_op(ef=3, n=48, k=3, s=1),
        mbv3_op(ef=6, n=96, k=3, s=2),
        mbv3_op(ef=6, n=96, k=3, s=1),
        mbv3_op(ef=6, n=96, k=3, s=1),
        op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=576),
        op(reduce_to_1x1, default_size=7, stride=1, padding='VALID'),
        op(slim.conv2d,
           stride=1,
           kernel_size=[1, 1],
           num_outputs=1024,
           normalizer_fn=None)
    ]))

# EdgeTPU friendly variant of MobilenetV3 that uses fused convolutions
# instead of depthwise in the early layers.
V3_EDGETPU = dict(
    defaults=dict(DEFAULTS),
    spec=[
        op(slim.conv2d, stride=2, num_outputs=32, kernel_size=(3, 3)),
        mbv3_fused(k=3, s=1, ef=1, n=16),
        mbv3_fused(k=3, s=2, ef=8, n=32),
        mbv3_fused(k=3, s=1, ef=4, n=32),
        mbv3_fused(k=3, s=1, ef=4, n=32),
        mbv3_fused(k=3, s=1, ef=4, n=32),
        mbv3_fused(k=3, s=2, ef=8, n=48),
        mbv3_fused(k=3, s=1, ef=4, n=48),
        mbv3_fused(k=3, s=1, ef=4, n=48),
        mbv3_fused(k=3, s=1, ef=4, n=48),
        mbv3_op(k=3, s=2, ef=8, n=96),
        mbv3_op(k=3, s=1, ef=4, n=96),
        mbv3_op(k=3, s=1, ef=4, n=96),
        mbv3_op(k=3, s=1, ef=4, n=96),
        mbv3_op(k=3, s=1, ef=8, n=96, residual=False),
        mbv3_op(k=3, s=1, ef=4, n=96),
        mbv3_op(k=3, s=1, ef=4, n=96),
        mbv3_op(k=3, s=1, ef=4, n=96),
        mbv3_op(k=5, s=2, ef=8, n=160),
        mbv3_op(k=5, s=1, ef=4, n=160),
        mbv3_op(k=5, s=1, ef=4, n=160),
        mbv3_op(k=5, s=1, ef=4, n=160),
        mbv3_op(k=3, s=1, ef=8, n=192),
        op(slim.conv2d, stride=1, num_outputs=1280, kernel_size=(1, 1)),
    ])


@slim.add_arg_scope
def mobilenet(input_tensor,
              num_classes=1001,
              depth_multiplier=1.0,
              scope='MobilenetV3',
              conv_defs=None,
              finegrain_classification_mode=False,
              **kwargs):
    """Creates mobilenet V3 network.

  Inference mode is created by default. To create training use training_scope
  below.

  with tf.contrib.slim.arg_scope(mobilenet_v3.training_scope()):
     logits, endpoints = mobilenet_v3.mobilenet(input_tensor)

  Args:
    input_tensor: The input tensor
    num_classes: number of classes
    depth_multiplier: The multiplier applied to scale number of
    channels in each layer.
    scope: Scope of the operator
    conv_defs: Which version to create. Could be large/small or
    any conv_def (see mobilenet_v3.py for examples).
    finegrain_classification_mode: When set to True, the model
    will keep the last layer large even for small multipliers. Following
    https://arxiv.org/abs/1801.04381
    it improves performance for ImageNet-type of problems.
      *Note* ignored if final_endpoint makes the builder exit earlier.
    **kwargs: passed directly to mobilenet.mobilenet:
      prediction_fn- what prediction function to use.
      reuse-: whether to reuse variables (if reuse set to true, scope
      must be given).
  Returns:
    logits/endpoints pair

  Raises:
    ValueError: On invalid arguments
  """
    if conv_defs is None:
        conv_defs = V3_LARGE
    if 'multiplier' in kwargs:
        raise ValueError('mobilenetv2 doesn\'t support generic '
                         'multiplier parameter use "depth_multiplier" instead.')
    if finegrain_classification_mode:
        conv_defs = copy.deepcopy(conv_defs)
        conv_defs['spec'][-1] = conv_defs['spec'][-1]._replace(
            multiplier_func=lambda params, multiplier: params)
    depth_args = {}
    with slim.arg_scope((lib.depth_multiplier,), **depth_args):
        return lib.mobilenet(
            input_tensor,
            conv_defs=conv_defs,
            scope=scope,
            multiplier=depth_multiplier,
            **kwargs)


mobilenet.default_image_size = 224
training_scope = lib.training_scope


@slim.add_arg_scope
def mobilenet_base(input_tensor, depth_multiplier=1.0, **kwargs):
    """Creates base of the mobilenet (no pooling and no logits) ."""
    return mobilenet(
        input_tensor, depth_multiplier=depth_multiplier, base_only=True, **kwargs)


def wrapped_partial(func, new_defaults=None,
                    **kwargs):
    """Partial function with new default parameters and updated docstring."""
    if not new_defaults:
        new_defaults = {}

    def func_wrapper(*f_args, **f_kwargs):
        new_kwargs = dict(new_defaults)
        new_kwargs.update(f_kwargs)
        return func(*f_args, **new_kwargs)

    functools.update_wrapper(func_wrapper, func)
    partial_func = functools.partial(func_wrapper, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


large = wrapped_partial(mobilenet, conv_defs=V3_LARGE)
small = wrapped_partial(mobilenet, conv_defs=V3_SMALL)
edge_tpu = wrapped_partial(mobilenet,
                           new_defaults={'scope': 'MobilenetEdgeTPU'},
                           conv_defs=V3_EDGETPU)
edge_tpu_075 = wrapped_partial(
    mobilenet,
    new_defaults={'scope': 'MobilenetEdgeTPU'},
    conv_defs=V3_EDGETPU,
    depth_multiplier=0.75,
    finegrain_classification_mode=True)

# Minimalistic model that does not have Squeeze Excite blocks,
# Hardswish, or 5x5 depthwise convolution.
# This makes the model very friendly for a wide range of hardware
large_minimalistic = wrapped_partial(mobilenet, conv_defs=V3_LARGE_MINIMALISTIC)
small_minimalistic = wrapped_partial(mobilenet, conv_defs=V3_SMALL_MINIMALISTIC)


def _reduce_consecutive_layers(conv_defs, start_id, end_id, multiplier=0.5):
    """Reduce the outputs of consecutive layers with multiplier.

  Args:
    conv_defs: Mobilenet conv_defs.
    start_id: 0-based index of the starting conv_def to be reduced.
    end_id: 0-based index of the last conv_def to be reduced.
    multiplier: The multiplier by which to reduce the conv_defs.

  Returns:
    Mobilenet conv_defs where the output sizes from layers [start_id, end_id],
    inclusive, are reduced by multiplier.

  Raises:
    ValueError if any layer to be reduced does not have the 'num_outputs'
    attribute.
  """
    defs = copy.deepcopy(conv_defs)
    for d in defs['spec'][start_id:end_id + 1]:
        d.params.update({
            'num_outputs': np.int(np.round(d.params['num_outputs'] * multiplier))
        })
    return defs


V3_LARGE_DETECTION = _reduce_consecutive_layers(V3_LARGE, 13, 16)
V3_SMALL_DETECTION = _reduce_consecutive_layers(V3_SMALL, 9, 12)


def compute_bottleneck(images, keep_probability=0.8, weight_decay=0.0, phase_train=True, reuse=None):
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
        return mobilenet(images, conv_defs=V3_LARGE)


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


__all__ = ['training_scope', 'mobilenet', 'V3_LARGE', 'V3_SMALL', 'large',
           'small', 'V3_LARGE_DETECTION', 'V3_SMALL_DETECTION']

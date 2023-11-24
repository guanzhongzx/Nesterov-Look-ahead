# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf
import pprint
import cifar10_input
import lookahead_optimizer
import radam_optimizer
import NALA_optimizer#########自己加的

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 256, ############# batch_size=256 ###############
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN ###NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL ###NUM_EXAMPLES_PER_EPOCH_FOR_EVAL=10000

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
# NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
NUM_EPOCHS_PER_DECAY = 20.0  # Epochs after which learning rate decays.

LEARNING_RATE_DECAY_FACTOR = 0.5  # Learning rate decay factor.
# INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)###替换字符串中的数字
  # tf.summary.histogram(tensor_name + '/activations', x)
  # tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  # with tf.device('/cpu:0'):
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  # if wd is not None:
  #   weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
  #   tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  images, labels = cifar10_input.distorted_inputs(batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  images, labels = cifar10_input.inputs(eval_data=eval_data, batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)###类型转换
  return images, labels


def inference(images):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64],##[卷积核的高度，卷积核的宽度，图像通道数3，卷积核个数]
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME') ##设置卷积步长，设置边界
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1 #归一化/标准化层
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.keras.layers.Flatten()(pool2)######### LeNet-5 的第三个卷积层有时写作Flatten？
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1 / 192.0, wd=None)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)
  ##########返回即logits
  return softmax_linear

def inference2(images):
    images = tf.image.resize_images(images, [32, 32])  ################
    # parameters = []###### AlexNet
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[6, 6, 3, 96],  ##[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
                                             stddev=1e-1,
                                             wd=None)
        conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')  ##设置卷积步长，设置边界
        biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        # parameters += [kernel, biases]
        _activation_summary(conv1)

    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 96, 256],  ##[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
                                             stddev=1e-1,
                                             wd=None)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        # parameters += [kernel, biases]
        _activation_summary(conv2)

    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool2')

    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 256, 384],  ##[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
                                             stddev=1e-1,
                                             wd=None)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)
        # parameters += [kernel, biases]
        _activation_summary(conv3)

    with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 384, 384],  ##[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
                                             stddev=1e-1,
                                             wd=None)
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope.name)
        # parameters += [kernel, biases]
        _activation_summary(conv4)

    with tf.variable_scope('conv5') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 384, 256],  ##[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
                                             stddev=1e-1,
                                             wd=None)
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope.name)
        # parameters += [kernel, biases]
        _activation_summary(conv5)

    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool5')

    # 第六层全连接层
    with tf.variable_scope('ful_con1') as scope:
      pool5 = tf.reshape(pool5, (-1, 3 * 3 * 256))
      weights = _variable_with_weight_decay('weights', shape=[3 * 3 * 256, 4096],
                                            stddev=0.1, wd=0.004)
      ful_bias1 = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.0))
      ful_con1 = tf.nn.relu(tf.add(tf.matmul(pool5, weights), ful_bias1), name=scope.name)
      _activation_summary(ful_con1)

    # 第七层第二层全连接层
    with tf.variable_scope('ful_con2') as scope:
      weights = _variable_with_weight_decay('weights', shape=[4096, 4096],
                                            stddev=0.1, wd=0.004)
      ful_bias2 = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.0))
      ful_con2 = tf.nn.relu(tf.add(tf.matmul(ful_con1, weights), ful_bias2), name=scope.name)
      _activation_summary(ful_con2)

    # # 第八层第三层全连接层
    # with tf.variable_scope('ful_con3') as scope:
    #   weights = _variable_with_weight_decay('weights', shape=[4096, 1000],
    #                                         stddev=0.1, wd=0.004)
    #   ful_bias3 = _variable_on_cpu('biases', [1000], tf.constant_initializer(0.0))
    #   ful_con3 = tf.nn.relu(tf.add(tf.matmul(ful_con2, weights), ful_bias3), name=scope.name)
    #   _activation_summary(ful_con3)

    # softmax层
    with tf.variable_scope('output_softmax') as scope:
      weights = _variable_with_weight_decay('weights', [4096, NUM_CLASSES],
                                              stddev=0.1, wd=None)
      biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
      output_softmax = tf.add(tf.matmul(ful_con2, weights), biases, name=scope.name)
      _activation_summary(output_softmax)
    return output_softmax #####, parameters

def inference3(images):
    images = tf.image.resize_images(images, [32, 32])  ################
    # parameters = []
    # with tf.variable_scope('conv1') as scope:
    #     kernel = _variable_with_weight_decay('weights',
    #                                          shape=[6, 6, 3, 96],  ##[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
    #                                          stddev=1e-1,
    #                                          wd=None)
    #     conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')  ##设置卷积步长，设置边界
    #     biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
    #     bias = tf.nn.bias_add(conv, biases)
    #     conv1 = tf.nn.relu(bias, name=scope.name)
    #     # parameters += [kernel, biases]
    #     _activation_summary(conv1)
    #第一层卷积层
    with tf.name_scope("conv1") as scope:
        #设置卷积核11×11,3通道,64个卷积核
        kernel1 = tf.Variable(tf.truncated_normal([6,6,3,96],mean=0,stddev=0.1,
                                                  dtype=tf.float32),name="weights")
        #卷积,卷积的横向步长和竖向补偿都为4
        conv = tf.nn.conv2d(images,kernel1,[1,2,2,1],padding="SAME")
        #初始化偏置
        biases = tf.Variable(tf.constant(0,shape=[96],dtype=tf.float32),trainable=True,name="biases")
        bias = tf.nn.bias_add(conv,biases)
        #RELU激活函数
        conv1 = tf.nn.relu(bias,name=scope)
        #统计参数
        # parameters += [kernel1,biases]
        #lrn处理
        lrn1 = tf.nn.lrn(conv1,4,bias=1,alpha=1e-3/9,beta=0.75,name="lrn1")
        #最大池化
        pool1 = tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID",name="pool1")

    #第二层卷积层
    with tf.name_scope("conv2") as scope:
        #初始化权重
        kernel2 = tf.Variable(tf.truncated_normal([3,3,96,256],dtype=tf.float32,stddev=0.1)
                              ,name="weights")
        conv = tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding="SAME")
        #初始化偏置
        biases = tf.Variable(tf.constant(0,dtype=tf.float32,shape=[256])
                             ,trainable=True,name="biases")
        bias = tf.nn.bias_add(conv,biases)
        #RELU激活
        conv2 = tf.nn.relu(bias,name=scope)
        # parameters += [kernel2,biases]
        #LRN
        lrn2 = tf.nn.lrn(conv2,4,1.0,alpha=1e-3/9,beta=0.75,name="lrn2")
        #最大池化
        pool2 = tf.nn.max_pool(lrn2,[1,3,3,1],[1,1,1,1],padding="VALID",name="pool2")

    #第三层卷积层
    with tf.name_scope("conv3") as scope:
        #初始化权重
        kernel3 = tf.Variable(tf.truncated_normal([3,3,256,384],dtype=tf.float32,stddev=0.1)
                              ,name="weights")
        conv = tf.nn.conv2d(pool2,kernel3,strides=[1,1,1,1],padding="SAME")
        biases = tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32),trainable=True,name="biases")
        bias = tf.nn.bias_add(conv,biases)
        #RELU激活层
        conv3 = tf.nn.relu(bias,name=scope)
        # parameters += [kernel3,biases]

    #第四层卷积层
    with tf.name_scope("conv4") as scope:
        #初始化权重
        kernel4 = tf.Variable(tf.truncated_normal([3,3,384,384],stddev=0.1,dtype=tf.float32),
                              name="weights")
        #卷积
        conv = tf.nn.conv2d(conv3,kernel4,strides=[1,1,1,1],padding="SAME")
        biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[384]),trainable=True,name="biases")
        bias = tf.nn.bias_add(conv,biases)
        #RELU激活
        conv4 = tf.nn.relu(bias,name=scope)
        # parameters += [kernel4,biases]

    #第五层卷积层
    with tf.name_scope("conv5") as scope:
        #初始化权重
        kernel5 = tf.Variable(tf.truncated_normal([3,3,384,256],stddev=0.1,dtype=tf.float32),
                              name="weights")
        conv = tf.nn.conv2d(conv4,kernel5,strides=[1,1,1,1],padding="SAME")
        biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[256]),name="biases")
        bias = tf.nn.bias_add(conv,biases)
        #REUL激活层
        conv5 = tf.nn.relu(bias)
        # parameters += [kernel5,bias]
        #最大池化
        pool5 = tf.nn.max_pool(conv5,[1,3,3,1],[1,1,1,1],padding="VALID",name="pool5")

    #第六层全连接层
    pool5 = tf.reshape(pool5,(-1,3*3*256))
    weight6 = tf.Variable(tf.truncated_normal([3*3*256,4096],stddev=0.1,dtype=tf.float32),
                           name="weight6")
    ful_bias1 = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[4096]),name="ful_bias1")
    ful_con1 = tf.nn.relu(tf.add(tf.matmul(pool5,weight6),ful_bias1))

    #第七层第二层全连接层
    weight7 = tf.Variable(tf.truncated_normal([4096,4096],stddev=0.1,dtype=tf.float32),
                          name="weight7")
    ful_bias2 = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[4096]),name="ful_bias2")
    ful_con2 = tf.nn.relu(tf.add(tf.matmul(ful_con1,weight7),ful_bias2))
    #
    #第八层第三层全连接层
    weight8 = tf.Variable(tf.truncated_normal([4096,1000],stddev=0.1,dtype=tf.float32),
                          name="weight8")
    ful_bias3 = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[1000]),name="ful_bias3")
    ful_con3 = tf.nn.relu(tf.add(tf.matmul(ful_con2,weight8),ful_bias3))

    #softmax层
    weight9 = tf.Variable(tf.truncated_normal([1000,NUM_CLASSES],stddev=0.1),dtype=tf.float32,name="weight9")
    bias9 = tf.Variable(tf.constant(0.0,shape=[NUM_CLASSES]),dtype=tf.float32,name="bias9")
    output_softmax = tf.nn.softmax(tf.matmul(ful_con3,weight9)+bias9)

    return output_softmax ###,parameters

def alexnet(images, keep_prob):
    images = tf.image.resize_images(images, [227, 227])  ################
    # conv1
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)

    # lrn1
    with tf.name_scope('lrn1') as scope:
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                  alpha=1e-4,
                                                  beta=0.75,
                                                  depth_radius=2,
                                                  bias=2.0)

    # pool1
    with tf.name_scope('pool1') as scope:
        pool1 = tf.nn.max_pool(lrn1,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID')

    # conv2
    with tf.name_scope('conv2') as scope:
        pool1_groups = tf.split(axis=3, value=pool1, num_or_size_splits=2)
        kernel = tf.Variable(tf.truncated_normal([5, 5, 48, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        kernel_groups = tf.split(axis=3, value=kernel, num_or_size_splits=2)
        conv_up = tf.nn.conv2d(pool1_groups[0], kernel_groups[0], [1, 1, 1, 1], padding='SAME')
        conv_down = tf.nn.conv2d(pool1_groups[1], kernel_groups[1], [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        biases_groups = tf.split(axis=0, value=biases, num_or_size_splits=2)
        bias_up = tf.nn.bias_add(conv_up, biases_groups[0])
        bias_down = tf.nn.bias_add(conv_down, biases_groups[1])
        bias = tf.concat(axis=3, values=[bias_up, bias_down])
        conv2 = tf.nn.relu(bias, name=scope)

    # lrn2
    with tf.name_scope('lrn2') as scope:
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                  alpha=1e-4,
                                                  beta=0.75,
                                                  depth_radius=2,
                                                  bias=2.0)

    # pool2
    with tf.name_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(lrn2,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID')

        # conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)

    # conv4
    with tf.name_scope('conv4') as scope:
        conv3_groups = tf.split(axis=3, value=conv3, num_or_size_splits=2)
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        kernel_groups = tf.split(axis=3, value=kernel, num_or_size_splits=2)
        conv_up = tf.nn.conv2d(conv3_groups[0], kernel_groups[0], [1, 1, 1, 1], padding='SAME')
        conv_down = tf.nn.conv2d(conv3_groups[1], kernel_groups[1], [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        biases_groups = tf.split(axis=0, value=biases, num_or_size_splits=2)
        bias_up = tf.nn.bias_add(conv_up, biases_groups[0])
        bias_down = tf.nn.bias_add(conv_down, biases_groups[1])
        bias = tf.concat(axis=3, values=[bias_up, bias_down])
        conv4 = tf.nn.relu(bias, name=scope)

    # conv5
    with tf.name_scope('conv5') as scope:
        conv4_groups = tf.split(axis=3, value=conv4, num_or_size_splits=2)
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        kernel_groups = tf.split(axis=3, value=kernel, num_or_size_splits=2)
        conv_up = tf.nn.conv2d(conv4_groups[0], kernel_groups[0], [1, 1, 1, 1], padding='SAME')
        conv_down = tf.nn.conv2d(conv4_groups[1], kernel_groups[1], [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        biases_groups = tf.split(axis=0, value=biases, num_or_size_splits=2)
        bias_up = tf.nn.bias_add(conv_up, biases_groups[0])
        bias_down = tf.nn.bias_add(conv_down, biases_groups[1])
        bias = tf.concat(axis=3, values=[bias_up, bias_down])
        conv5 = tf.nn.relu(bias, name=scope)

    # pool5
    with tf.name_scope('pool5') as scope:
        pool5 = tf.nn.max_pool(conv5,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID', )

    # flattened6
    with tf.name_scope('flattened6') as scope:
        flattened = tf.reshape(pool5, shape=[-1, 6 * 6 * 256])

    # fc6
    with tf.name_scope('fc6') as scope:
        weights = tf.Variable(tf.truncated_normal([6 * 6 * 256, 4096],
                                                  dtype=tf.float32,
                                                  stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.xw_plus_b(flattened, weights, biases)
        fc6 = tf.nn.relu(bias)

    # dropout6
    with tf.name_scope('dropout6') as scope:
        dropout6 = tf.nn.dropout(fc6, keep_prob)

    # fc7
    with tf.name_scope('fc7') as scope:
        weights = tf.Variable(tf.truncated_normal([4096, 4096],
                                                  dtype=tf.float32,
                                                  stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.xw_plus_b(dropout6, weights, biases)
        fc7 = tf.nn.relu(bias)

    # dropout7
    with tf.name_scope('dropout7') as scope:
        dropout7 = tf.nn.dropout(fc7, keep_prob)

    # fc8
    with tf.name_scope('fc8') as scope:
        weights = tf.Variable(tf.truncated_normal([4096, NUM_CLASSES],
                                                  dtype=tf.float32,
                                                  stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[NUM_CLASSES], dtype=tf.float32),
                             trainable=True, name='biases')
        fc8 = tf.nn.xw_plus_b(dropout7, weights, biases)

    return fc8

def alexnet2(images, keep_prob):
    images = tf.image.resize_images(images, [32, 32])  ################name_scope#variable_scope
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[6, 6, 3, 96],  ##[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
                                             stddev=1e-1,
                                             wd=None)
        conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')  ##设置卷积步长，设置边界
        biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        # _activation_summary(conv1)

    # lrn1
    with tf.name_scope('lrn1') as scope:
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                  alpha=1e-4,
                                                  beta=0.75,
                                                  depth_radius=2,
                                                  bias=2.0)

    # pool1
    with tf.name_scope('pool1') as scope:
        pool1 = tf.nn.max_pool(lrn1,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID')

    # conv2
    with tf.variable_scope('conv2') as scope:
        pool1_groups = tf.split(axis=3, value=pool1, num_or_size_splits=2)
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 48, 256],  ##[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
                                             stddev=1e-1,
                                             wd=None)
        kernel_groups = tf.split(axis=3, value=kernel, num_or_size_splits=2)
        conv_up = tf.nn.conv2d(pool1_groups[0], kernel_groups[0], [1, 1, 1, 1], padding='SAME')
        conv_down = tf.nn.conv2d(pool1_groups[1], kernel_groups[1], [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        biases_groups = tf.split(axis=0, value=biases, num_or_size_splits=2)
        bias_up = tf.nn.bias_add(conv_up, biases_groups[0])
        bias_down = tf.nn.bias_add(conv_down, biases_groups[1])
        bias = tf.concat(axis=3, values=[bias_up, bias_down])
        conv2 = tf.nn.relu(bias, name=scope.name)
        # _activation_summary(conv2)

    # lrn2
    with tf.name_scope('lrn2') as scope:
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                  alpha=1e-4,
                                                  beta=0.75,
                                                  depth_radius=2,
                                                  bias=2.0)

    # pool2
    with tf.name_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(lrn2,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 1, 1, 1],
                               padding='VALID')

        # conv3
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 256, 384],  ##[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
                                             stddev=1e-1,
                                             wd=None)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)
        # _activation_summary(conv3)

    # conv4
    with tf.variable_scope('conv4') as scope:
        conv3_groups = tf.split(axis=3, value=conv3, num_or_size_splits=2)
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 192, 384],  ##[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
                                             stddev=1e-1,
                                             wd=None)
        kernel_groups = tf.split(axis=3, value=kernel, num_or_size_splits=2)
        conv_up = tf.nn.conv2d(conv3_groups[0], kernel_groups[0], [1, 1, 1, 1], padding='SAME')
        conv_down = tf.nn.conv2d(conv3_groups[1], kernel_groups[1], [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.0))
        biases_groups = tf.split(axis=0, value=biases, num_or_size_splits=2)
        bias_up = tf.nn.bias_add(conv_up, biases_groups[0])
        bias_down = tf.nn.bias_add(conv_down, biases_groups[1])
        bias = tf.concat(axis=3, values=[bias_up, bias_down])
        conv4 = tf.nn.relu(bias, name=scope.name)
        # _activation_summary(conv4)

    # conv5
    with tf.variable_scope('conv5') as scope:
        conv4_groups = tf.split(axis=3, value=conv4, num_or_size_splits=2)
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 192, 256],  ##[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
                                             stddev=1e-1,
                                             wd=None)
        kernel_groups = tf.split(axis=3, value=kernel, num_or_size_splits=2)
        conv_up = tf.nn.conv2d(conv4_groups[0], kernel_groups[0], [1, 1, 1, 1], padding='SAME')
        conv_down = tf.nn.conv2d(conv4_groups[1], kernel_groups[1], [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        biases_groups = tf.split(axis=0, value=biases, num_or_size_splits=2)
        bias_up = tf.nn.bias_add(conv_up, biases_groups[0])
        bias_down = tf.nn.bias_add(conv_down, biases_groups[1])
        bias = tf.concat(axis=3, values=[bias_up, bias_down])
        conv5 = tf.nn.relu(bias, name=scope.name)
        # _activation_summary(conv5)

    # pool5
    with tf.name_scope('pool5') as scope:
        pool5 = tf.nn.max_pool(conv5,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 1, 1, 1],
                               padding='VALID', )

    # flattened6
    with tf.name_scope('flattened6') as scope:
        flattened = tf.reshape(pool5, shape=[-1, 3 * 3 * 256])

    # fc6
    with tf.variable_scope('fc6') as scope:
        weights = _variable_with_weight_decay('weights', shape=[3 * 3 * 256, 4096],
                                              stddev=0.1, wd=0.004)
        biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.0))
        bias = tf.nn.xw_plus_b(flattened, weights, biases)
        fc6 = tf.nn.relu(bias, name=scope.name)
        # _activation_summary(fc6)

    # dropout6
    with tf.name_scope('dropout6') as scope:
        dropout6 = tf.nn.dropout(fc6, keep_prob)

    # fc7
    with tf.variable_scope('fc7') as scope:
        weights = _variable_with_weight_decay('weights', shape=[4096, 4096],
                                              stddev=0.1, wd=0.004)
        biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.0))
        bias = tf.nn.xw_plus_b(dropout6, weights, biases)
        fc7 = tf.nn.relu(bias, name=scope.name)
        # _activation_summary(fc7)

    # dropout7
    with tf.name_scope('dropout7') as scope:
        dropout7 = tf.nn.dropout(fc7, keep_prob)

    # fc8
    with tf.variable_scope('fc8') as scope:
        weights = _variable_with_weight_decay('weights', shape=[4096, NUM_CLASSES],
                                              stddev=0.1, wd=0.004)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        fc8 = tf.nn.xw_plus_b(dropout7, weights, biases, name=scope.name)
        # _activation_summary(fc8)

    return fc8

def softmax_summaries(loss, logits, one_hot_labels, name="softmax"):
  """Create the softmax summaries for this cross entropy loss.

  Args:
    logits: The [batch_size, classes] float tensor representing the logits.
    one_hot_labels: The float tensor representing actual class ids. If this is
      [batch_size, classes], then we take the argmax of it first.
    name: Prepended to summary scope.
  """
  tf.summary.scalar(name + "_loss", loss)

  one_hot_labels = tf.cond(
      tf.equal(tf.rank(one_hot_labels),
               2), lambda: tf.to_int32(tf.argmax(one_hot_labels, 1)),
      lambda: tf.to_int32(one_hot_labels))

  in_top_1 = tf.nn.in_top_k(logits, one_hot_labels, 1)
  tf.summary.scalar(name + "_precision_1",
                    tf.reduce_mean(tf.to_float(in_top_1)))
  in_top_5 = tf.nn.in_top_k(logits, one_hot_labels, 5)
  tf.summary.scalar(name + "_precision_5",
                    tf.reduce_mean(tf.to_float(in_top_5)))


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')#######损失函数为交叉熵
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)
  softmax_summaries(cross_entropy_mean, logits, labels)
  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def accuracy(logits, labels):
    labels = tf.cast(labels, tf.int64)
    accuracy_p = tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32)
    accuracy_mean = tf.reduce_mean(accuracy_p)
    tf.add_to_collection('accuracies', accuracy_mean)
    return tf.add_n(tf.get_collection('accuracies'), name='total_accuracy')
    ################# test ###############自己加的测试准确率##############

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg') ##设置超参数：1-衰减率
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # # Attach a scalar summary to all individual losses and the total loss; do the
  # # same for the averaged version of the losses.
  # for l in losses + [total_loss]:
  #   # Name each loss as '(raw)' and name the moving average version of the loss
  #   # as the original loss name.
  #   tf.summary.scalar(l.op.name + ' (raw)', l)
  #   tf.summary.scalar(l.op.name, loss_averages.average(l))
  return loss_averages_op

def convergence_rate():
    return tf.add_n(tf.get_collection('convergence_rate'), name='total_convergence')

def train(total_loss, global_step, optimizer='sgd', lr=0.1): ##设置默认优化算法，设置默认学习率
    #############def train(total_loss, total_accuracy, global_step, optimizer='sgd', lr=0.1):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size ###一轮训练(epoch)的step数50000/256
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)###衰减一次经过(50000/256)*20=3906个step
  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(lr,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,#调用学习率的衰减因子0.5
                                  staircase=True)
  # warm_up_step = 1000.0
  # lr = lr * tf.cast(tf.minimum(1.0, (warm_up_step / tf.cast(global_step, tf.float32)) ** 2), tf.float32)

  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)
  ###acc = total_accuracy ###########

  #pprint.pprint(total_loss) ############测 试###########

  OptimizerDict = {'sgd': tf.train.GradientDescentOptimizer,
                   'adam': tf.train.AdamOptimizer,
                   'NAG': tf.train.MomentumOptimizer,
                   'adadelta': tf.train.AdadeltaOptimizer,
                   'radam': radam_optimizer.RAdamOptimizer,
                   'NALAadam': tf.train.AdamOptimizer,
                   'lookaheadadam': tf.train.AdamOptimizer,
                   'lookaheadradam': radam_optimizer.RAdamOptimizer,
                   }
  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    if optimizer.find('NAG') >= 0: ###确认是NALA算法，返回0，否则-1
      opt = OptimizerDict[optimizer](lr, momentum=0.1, use_nesterov=True) ###
    else:
      opt = OptimizerDict[optimizer](lr) ###设置内循环优化器
    if optimizer.find('NALA') >= 0: ###确认是NALA算法
      opt = NALA_optimizer.LookaheadOptimizer(opt, interval_steps=5, alpha=-0.3) ###调用LO(内循环优化器，迭代数，步长)
    if optimizer.find('lookahead') >= 0: ###确认是Lookahead算法
      opt = lookahead_optimizer.LookaheadOptimizer(opt, interval_steps=5, alpha=0.5) ###调用LO(内循环优化器，迭代数，步长)
    grads = opt.compute_gradients(total_loss)###Returns a list of (gradient, variable) pairs 计算梯度和参数，即grads_and_vars

    var_list = [v for g, v in grads if g is not None]  #更新前参数
    theta = [tf.norm(v) for v in var_list]
    norm_last = tf.norm(theta)  ##计算范数##print('theta= ', norm)

  # Apply gradients.
  if optimizer.find('NALA') >= 0:#####################
    apply_gradient_op = opt.apply_gradients(total_loss, grads, global_step=global_step)#lookahead更新参数##########
  else:
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)  #其他优化器更新参数
  # print('apply_gradient_op: ', apply_gradient_op)##############

  with tf.control_dependencies([apply_gradient_op]):
    theta = [tf.norm(var) for var in tf.trainable_variables()]##更新后参数
    norm = tf.norm(theta)###参数的范数
    convergence = 1 - (norm / norm_last)  ## print('convergence= ', convergence)####收敛速率指标
    tf.add_to_collection('convergence_rate', convergence)
  # # Add histograms for trainable variables.
  # for var in tf.trainable_variables():
  #   tf.summary.histogram(var.op.name, var)

  # # Add histograms for gradients.
  # for grad, var in grads:
  #   if grad is not None:
  #     tf.summary.histogram(var.op.name + '/gradients', grad)

  # # Track the moving averages of all trainable variables.
  # variable_averages = tf.train.ExponentialMovingAverage(
  #     MOVING_AVERAGE_DECAY, global_step)
  # with tf.control_dependencies([apply_gradient_op]):
  #   variables_averages_op = variable_averages.apply(tf.trainable_variables())

  return apply_gradient_op

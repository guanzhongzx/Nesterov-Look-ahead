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

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from datetime import datetime

import tensorflow as tf
import cifar10

import pickle ##########画图用的

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('optimizer', 'NALAadam',####在cifar10中选择优化算法###NALAadam###lookaheadadam
                           """optimizer e.g. sgd/NALAadam/adam/radam/lookaheadxx""")
tf.app.flags.DEFINE_float('learning_rate', 0.001,##########设置内循环优化器学习率0.1/0.001
                          """init learning rate""")
tf.app.flags.DEFINE_integer('max_steps', 45000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,####False###
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """How often to log results to the console.""")

tf.app.flags.DEFINE_string('eval_data', 'test',############读测试数据
                            """Data in tfds.Split.TEST/TRAIN for inputs.""")####多行注释用“contrl+/”

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()
    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):###指定在第1块cpu上运行
      images, labels = cifar10.distorted_inputs() ###失真处理
      imagesE, labelsE = cifar10.inputs(FLAGS.eval_data)  #######################test
    # Build a Graph that computes the logits predictions from the
    # inference model.

    logits = cifar10.alexnet2(images, keep_prob=0.5)#logits = cifar10.inference(images)##网络输出推测##
    # logits2 = cifar10.inference(imagesE)#

    # Calculate loss.
    loss = cifar10.loss(logits, labels)##计算交叉熵损失值##
    accuracy = cifar10.accuracy(logits, labels)  ########自己加的测试准确率#########

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step, optimizer=FLAGS.optimizer, lr=FLAGS.learning_rate)
    convergence = cifar10.convergence_rate()

    RV=[]
    # 更新模型参数,代入FLAGS设定的优化器和学习率
    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs([loss, accuracy, convergence])# Asks for loss value.###计算loss###Represents arguments to be added to a Session.run() call.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time
          # print(run_context)#####################
          print('run_values: ', run_values)#######################执行打印run_values#

          loss_value, acc, con= run_values.results#########loss_value = run_values.results###########
          RV.append(run_values.results)
          print('accuracy: ', acc)
          print('convergence_rate: ', con)
          file_name = './tmp/run_values/' + 'LOSSandACCURACY.pkl'########保存路径
          with open(file_name, 'wb') as fp:
              pickle.dump(RV, fp) ##########存数据

          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration ###在cifar10.py中batch_size=256
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print(format_str % (datetime.now(), self._step, loss_value,
                              examples_per_sec, sec_per_batch))


    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),###挂起于loss=Nan
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)###########执行训练
       ####### mon_sess.run(accuracy)############ mon_sess.run(loss)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):###判断目录和文件是否存在
      tf.gfile.DeleteRecursively(FLAGS.train_dir)###递归删除所有目录及其文件
  #     restore_mode_pb(FLAGS.train_dir)
  # else:
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()
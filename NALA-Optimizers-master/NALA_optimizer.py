# ==============================================================================
# Author: Feiteng Li
# ==============================================================================

"""LookAhead for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer

import tensorflow as tf #########添加

class LookaheadOptimizer(optimizer.Optimizer): ###派生类(基类)# 派生类继承基类之属性
  """Wrapper optimizer that implements the Lookahead Optimizer.

  See [Michael et al., 2019](https://arxiv.org/abs/1907.08610)
  ([pdf](https://arxiv.org/pdf/1907.08610.pdf)).
  """

  def __init__(self,
               opt,
               interval_steps=5,  # k 内循环步数
               alpha= -0.5, ###NALA外循环参数衰减因子
               beta= 0.001, #####NALA外循环迭代步长
               name="Lookahead"):
    """Construct a new model average optimizer.

    Args:
      opt: The actual optimizer that will be used to update local variables
      interval_steps: An int point value to controls the frequency of the
        update of slow variables
      name: string. Optional name of the returned operation
    """
    super(LookaheadOptimizer, self).__init__(opt._use_locking, name)
    self._opt = opt#####内循环优化器
    self._interval_steps = interval_steps
    self._alpha = alpha
    self._beta = beta

  def _get_step_accumulator(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return self._get_non_slot_variable("step", graph=graph)

  def compute_gradients(self, *args, **kwargs):
    """Compute gradients of "loss" for the variables in "var_list".

    This simply wraps the compute_gradients() from the real optimizer.

    Args:
      *args: Arguments for compute_gradients().
      **kwargs: Keyword arguments for compute_gradients().

    Returns:
      A list of (gradient, variable) pairs.
    """
    return self._opt.compute_gradients(*args, **kwargs)

  def apply_gradients(self, total_loss, grads_and_vars, global_step=None, name=None):
  # def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables.

    This contains most of the synchronization implementation and also wraps the
    apply_gradients() from the real optimizer. The chief work updates global
    variables.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      A conditional 'Operation' that update both local and global variables or
      just local variables

    Raises:
      ValueError: If the grads_and_vars is empty.
      ValueError: If global step is not provided, the staleness cannot be
        checked.
    """
    if not grads_and_vars:
      raise ValueError("Must supply at least one variable")
    if global_step is None:
      print("{} WARNING: Global step is None.".format(self._name))

    var_list = [v for g, v in grads_and_vars if g is not None]#网络参数列表
    # print('grads_and_vars: ', grads_and_vars)
    # print('var_list:', var_list)
    # grad_list = [g for g, v in grads_and_vars if g is not None]######
    # grad_list = tf.clip_by_norm(grads_and_vars, clip_norm=0.5)####axes=1, ####
    # grads_and_vars = (grad_list, var_list)
    # print('grad_list:', grad_list)

    first_var = min(var_list, key=lambda x: x.name)###取名字最小的元素/一层网络(conv1/biases:0)？#print('first_var:', first_var)

    self._create_non_slot_variable(initial_value=0,
                                   name="step",
                                   colocate_with=first_var)######创建一个名为step的与插槽无关的变量
    # Create slots for local vars.
    for var in var_list:
      self._zeros_slot(var, 'local', self._name)
    #for grad in grad_list:##########for var in var_list:
    #  self._zeros_slot(grad, 'grad_local', self._name)############self._zeros_slot(var, 'local', self._name)

    apply_updates = self._opt.apply_gradients(grads_and_vars, global_step=global_step)####调用内循环优化器并更新参数！！！
    with ops.control_dependencies([apply_updates]):
      local_update = state_ops.assign_add(self._get_step_accumulator(), 1, name="local_step_update").op#step+1

    def _update_variables():#################
      global_assignments = [] #参数更新操作的列表
      local_assignments = [] #参数更新操作的列表
      Lookpoint = []

      for (grad, var) in grads_and_vars: ########### grads_and_vars包含梯度值和参数值
        if grad is None or var is None:
          continue

        local = self.get_slot(var, 'local')#######获取外循环参数
        #grad_local = self.get_slot(grad, 'grad_local')  #######
        # print('var: ', var)###
        # print('local: ', local)

        #next_var = (1.0 - self._alpha) * local + self._alpha * var ##########参数更新 slow weights = 外local + alpha*(内var-外local)

        yt = (1.0 + self._alpha) * var - self._alpha * local  ##########用动量估计参数## yt = var + alpha*(var-local)

        Lookpoint.append(state_ops.assign(var, yt, use_locking=self._opt._use_locking))
        # global_assignments.append(state_ops.assign(var, yt, use_locking=self._opt._use_locking))#逐层更新fast weights
        # local_assignments.append(state_ops.assign(local, yt, use_locking=self._opt._use_locking))#逐层更新slow weights

      # next_grads_and_vars = self._opt.compute_gradients(total_loss)
      with ops.control_dependencies(Lookpoint):
        next_grads_and_vars = self._opt.compute_gradients(total_loss)
      # print('next_grads: ', next_grads_and_vars)
      # ng_list = [ng for ng, nv in next_grads_and_vars if ng is not None]
      # print('ng_list: ', ng_list)

      for ((grad, var), (n_g, n_v)) in zip(grads_and_vars, next_grads_and_vars): ###########
        if grad is None or var is None:
          continue

        local = self.get_slot(var, 'local')  #######获取外循环参数
        next_var = n_v - self._beta * n_g ##########参数alpha待定???grad应为估计的下一步梯度

        global_assignments.append(state_ops.assign(var, next_var, use_locking=self._opt._use_locking))#逐层更新fast weights
        local_assignments.append(state_ops.assign(local, next_var, use_locking=self._opt._use_locking))#逐层更新slow weights

      with ops.control_dependencies(global_assignments):
        # update local variables.
        return control_flow_ops.group(*(local_assignments))
        #group(*inputs, **kwargs): When this op finishes, all ops in `inputs` have finished. This op has no output.

    with ops.control_dependencies([local_update]):
      condition = math_ops.equal(math_ops.mod(self._get_step_accumulator(), self._interval_steps), 0)###step/5?=0判断是否更新外循环参数
      conditional_update = control_flow_ops.cond(condition,
                                                 true_fn=_update_variables,
                                                 false_fn=control_flow_ops.no_op)

    with ops.control_dependencies([conditional_update]):
      return control_flow_ops.no_op()

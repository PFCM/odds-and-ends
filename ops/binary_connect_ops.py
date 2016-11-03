import os

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import common_shapes


_binary_connect_module = tf.load_op_library(
    os.path.join(os.path.dirname(__file__), 'binary_connect.so'))
binary_connect = _binary_connect_module.binary_connect


tf.RegisterShape("BinaryConnect")(common_shapes.call_cpp_shape_fn)


@ops.RegisterGradient("BinaryConnect")
def binary_connect_grad(op, grad):
    """"Gradients" for binary connect. They are in fact not the gradients,
    the gradients are zero. However, for the purposes of backpropagation
    we are prepared to suspend disbelief and pretend that the step function
    is in fact a hard tanh.

    Args:
        op: the `binary_connect` `Operation` we are "differentiating."
        grad: grad with respect to output.

    Returns:
        pseudo-gradients with respect to input of `binary_connect`.
    """
    return grad * tf.cast(tf.abs(op.inputs[0]) < 1.0, tf.float32)

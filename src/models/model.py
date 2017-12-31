import tensorflow as tf
import numpy as np


# utility functions for weight and bias init
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# conv and pool operations
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# tensorboard + layer utilities
def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


def fc_layer(input_tensor, num_units, layer_name, keep_prob_tensor=None, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.

    Here we read the shape of the incoming tensor (flatten if needed) and use it
    to set shapes of variables
    """

    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # reshape input_tensor if needed
        input_shape = input_tensor.get_shape()
        if len(input_shape) == 4:
            ndims = np.int(np.product(input_shape[1:]))
            input_tensor = tf.reshape(input_tensor, [-1, ndims])
        elif len(input_shape) == 2:
            ndims = input_shape[-1].value
        else:
            raise RuntimeError('Strange input tensor shape: {}'.format(input_shape))
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([ndims, num_units])
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = bias_variable([num_units])
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            pre_activate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram(layer_name + '/pre_activations', pre_activate)
        activations = act(pre_activate, 'activation')
        tf.summary.histogram(layer_name + '/activations', activations)
        if keep_prob_tensor is not None:
            activations_drop = tf.nn.dropout(activations, keep_prob_tensor)
            return activations_drop
    return activations


def conv_pool_layer(input_tensor, filter_size, num_filters, layer_name, act=tf.nn.relu, pool=True):
    """Reusable code for making a simple conv_pool layer.

    It does a 2D convolution, bias add, and then uses relu to nonlinearize, (optionally) followed by 2x2 max pooling
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.

    Here we read the shape of the incoming tensor and use it to set shapes of variables
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        patches_in = input_tensor.get_shape()[-1].value

        with tf.name_scope('weights'):
            weights = weight_variable([filter_size, filter_size, patches_in, num_filters])
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = bias_variable([num_filters])
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = conv2d(input_tensor, weights) + biases
            # tf.summary.histogram(layer_name + '/pre_activations', preactivate)
        activations = act(preactivate, 'activation')
        # tf.summary.histogram(layer_name + '/activations', activations)
        if pool:
            pooled_activations = max_pool_2x2(activations)
            # tf.summary.histogram(layer_name + '/pooled_activations', pooled_activations)

            return pooled_activations
        else:
            return activations


# MODEL
def conv_net(x_input, categories=200, keep_prob_=None):
    out_1 = conv_pool_layer(x_input, filter_size=3, num_filters=16, layer_name='conv_1', pool=False)
    out_2 = conv_pool_layer(out_1, filter_size=3, num_filters=16, layer_name='conv_pool_2')
    out_3 = conv_pool_layer(out_2, filter_size=3, num_filters=16, layer_name='conv_3', pool=False)
    out_4 = conv_pool_layer(out_3, filter_size=3, num_filters=32, layer_name='conv_pool_4')
    out_5 = conv_pool_layer(out_4, filter_size=3, num_filters=32, layer_name='conv_pool_5')
    out_6 = conv_pool_layer(out_5, filter_size=3, num_filters=64, layer_name='conv_pool_6', pool=False)
    out_7 = fc_layer(out_6, num_units=128, layer_name='FC_1', keep_prob_tensor=keep_prob_)
    out_8 = fc_layer(out_7, num_units=256, layer_name='FC_2', keep_prob_tensor=keep_prob_)
    logits_ = fc_layer(out_8, num_units=categories, layer_name='logits', act=tf.identity)

    return logits_


if __name__ == "__main__":
    x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])

    logits = conv_net(x)

    print(logits.get_shape())

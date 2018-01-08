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
    x_input = tf.cast(x_input, tf.float32)
    x_input = (x_input - 128.0) / 128.0

    out_1 = conv_pool_layer(x_input, filter_size=3, num_filters=16, layer_name='conv_1', pool=False)
    print(out_1.shape)
    out_2 = conv_pool_layer(out_1, filter_size=3, num_filters=16, layer_name='conv_pool_2')
    print(out_2.shape)
    out_3 = conv_pool_layer(out_2, filter_size=3, num_filters=16, layer_name='conv_3', pool=False)
    print(out_3.shape)
    out_4 = conv_pool_layer(out_3, filter_size=3, num_filters=32, layer_name='conv_pool_4')
    print(out_4.shape)
    out_5 = conv_pool_layer(out_4, filter_size=3, num_filters=32, layer_name='conv_pool_5')
    print(out_5.shape)
    out_6 = conv_pool_layer(out_5, filter_size=3, num_filters=64, layer_name='conv_pool_6', pool=False)
    print(out_6.shape)
    out_7 = fc_layer(out_6, num_units=128, layer_name='FC_1', keep_prob_tensor=keep_prob_)
    print(out_7.shape)
    out_8 = fc_layer(out_7, num_units=256, layer_name='FC_2', keep_prob_tensor=keep_prob_)
    print(out_8.shape)
    logits_ = fc_layer(out_8, num_units=categories, layer_name='logits', act=tf.identity)

    return logits_


def conv_2d(inputs, filters, kernel_size, name=None):
    """3x3 conv layer: ReLU + (1, 1) stride + He initialization"""

    # He initialization = normal dist with stdev = sqrt(2.0/fan-in)
    stddev = np.sqrt(2 / (np.prod(kernel_size) * int(inputs.shape[3])))
    out = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size,
                           padding='same', activation=tf.nn.relu,
                           kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
                           name=name)
    tf.summary.histogram('act' + name, out)

    return out


def dense_relu(inputs, units, name=None):
    """3x3 conv layer: ReLU + He initialization"""

    # He initialization: normal dist with stdev = sqrt(2.0/fan-in)
    stddev = np.sqrt(2 / int(inputs.shape[1]))
    out = tf.layers.dense(inputs, units, activation=tf.nn.relu,
                          kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
                          name=name)

    tf.summary.histogram('act' + name, out)

    return out


def dense(inputs, units, name=None):
    """3x3 conv layer: ReLU + He initialization"""

    # He initialization: normal dist with stdev = sqrt(2.0/fan-in)
    stddev = np.sqrt(2 / int(inputs.shape[1]))
    out = tf.layers.dense(inputs, units,
                          kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
                          name=name)
    tf.summary.histogram('act' + name, out)

    return out


def vgg_16(training_batch, dropout_keep_prob):
    """VGG-like conv-net
    Args:
    training_batch: batch of images (N, 56, 56, 3)
    config: training configuration object
    Returns:
    class prediction scores
    """
    img = tf.cast(training_batch, tf.float32)
    out = (img - 128.0) / 128.0

    tf.summary.histogram('img', training_batch)
    # (N, 56, 56, 3)
    out = conv_2d(out, 64, (3, 3), 'conv1_1')
    out = conv_2d(out, 64, (3, 3), 'conv1_2')
    out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), name='pool1')

    # (N, 28, 28, 64)
    out = conv_2d(out, 128, (3, 3), 'conv2_1')
    out = conv_2d(out, 128, (3, 3), 'conv2_2')
    out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), name='pool2')

    # (N, 14, 14, 128)
    out = conv_2d(out, 256, (3, 3), 'conv3_1')
    out = conv_2d(out, 256, (3, 3), 'conv3_2')
    out = conv_2d(out, 256, (3, 3), 'conv3_3')
    out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), name='pool3')

    # (N, 7, 7, 256)
    out = conv_2d(out, 512, (3, 3), 'conv4_1')
    out = conv_2d(out, 512, (3, 3), 'conv4_2')
    out = conv_2d(out, 512, (3, 3), 'conv4_3')

    # fc1: flatten -> fully connected layer
    # (N, 7, 7, 512) -> (N, 25088) -> (N, 4096)
    out = tf.contrib.layers.flatten(out)
    out = dense_relu(out, 4096, 'fc1')
    out = tf.nn.dropout(out, dropout_keep_prob)

    # fc2
    # (N, 4096) -> (N, 2048)
    out = dense_relu(out, 2048, 'fc2')
    out = tf.nn.dropout(out, dropout_keep_prob)

    # softmax
    # (N, 2048) -> (N, 200)
    logits = dense(out, 200, 'fc3')

    return logits


if __name__ == "__main__":
    x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])

    logits = conv_net(x)

    print(logits.get_shape())

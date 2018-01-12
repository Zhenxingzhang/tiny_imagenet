import tensorflow as tf
import numpy as np


def conv_2d_relu(inputs, filters, kernel_size, name=None):
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


def mnist_layer(training_batch, categories, dropout_keep_prob):
    """VGG-like conv-net
    Args:
    training_batch: batch of images (N, 56, 56, 3)
    config: training configuration object
    Returns:
    class prediction scores
    """
    out = tf.cast(training_batch, tf.float32)
    out = tf.reshape(out, [-1, 28, 28, 1])
    tf.summary.histogram('img', training_batch)

    # (N, 56, 56, 3)
    out = conv_2d_relu(out, 64, (3, 3), 'conv1_2')
    out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), name='pool1')

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
    logits = dense(out, categories, 'fc3')

    return logits


if __name__ == "__main__":
    x_input = tf.placeholder(tf.int8, shape=[None, 28, 28, 1])

    logits = mnist_layer(x_input, 10, 1.0)

    print(logits.shape)

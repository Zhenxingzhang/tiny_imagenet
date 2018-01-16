import tensorflow as tf
import numpy as np

def conv_2d_relu(inputs, filters, kernel_size, name=None):
    """3x3 conv layer: ReLU + (1, 1) stride + He initialization"""

    # He initialization = normal dist with stdev = sqrt(2.0/fan-in)
    stddev = np.sqrt(2 / (np.prod(kernel_size) * int(inputs.shape[3])))

    out = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size,
                           padding='same',
                           kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                           # kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
                           name=name)
    out = tf.layers.batch_normalization(out, training=True)
    out = tf.nn.relu(out)

    tf.summary.histogram('act' + name, out)

    return out


def dense_relu(inputs, units, name=None):
    """3x3 conv layer: ReLU + He initialization"""

    # He initialization: normal dist with stdev = sqrt(2.0/fan-in)
    stddev = np.sqrt(2 / int(inputs.shape[1]))
    out = tf.layers.dense(inputs, units,
                          kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
                          name=name)
    out = tf.layers.batch_normalization(out, training=True)
    out = tf.nn.relu(out)

    tf.summary.histogram('act' + name, out)

    return out


def dense(inputs, units, name=None):
    """3x3 conv layer: ReLU + He initialization"""

    # He initialization: normal dist with stdev = sqrt(2.0/fan-in)
    stddev = np.sqrt(2 / int(inputs.shape[1]))
    out = tf.layers.dense(inputs, units,
                          kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
                          name=name)
    out = tf.layers.batch_normalization(out, training=True)
    tf.summary.histogram('act' + name, out)

    return out


def conv_net_3(training_batch, categories, dropout_prob):
    out = tf.cast(training_batch, tf.float32)
    out = (out - 128.0) / 128.0
    tf.summary.histogram('img', training_batch)

    out_1 = conv_2d_relu(out, 16, [3, 3], name='conv_1')
    out_2 = conv_2d_relu(out_1, 16, [3, 3], name='conv_2')
    pool_1 = tf.layers.max_pooling2d(out_2, (2, 2), (2, 2), name='pool1')

    out_3 = conv_2d_relu(pool_1, 16, [3, 3], name='conv_3')
    out_4 = conv_2d_relu(out_3, 16, [3, 3], name='conv_4')
    pool_2 = tf.layers.max_pooling2d(out_4, (2, 2), (2, 2), name='pool2')

    out_5 = conv_2d_relu(pool_2, 16, [3, 3], name='conv_5')
    pool_3 = tf.layers.max_pooling2d(out_5, (2, 2), (2, 2), name='pool3')

    out_6 = conv_2d_relu(pool_3, 16, [3, 3], name='conv_6')

    flat_1 = tf.contrib.layers.flatten(out_6)
    dense_1 = dense_relu(flat_1, 1024, 'fc1')
    dropout_1 = tf.nn.dropout(dense_1, dropout_prob)

    dense_2 = dense_relu(dropout_1, 512, 'fc2')
    dropout_2 = tf.nn.dropout(dense_2, dropout_prob)

    logits = dense(dropout_2, categories, 'fc3')
    return logits


def vgg_16(training_batch, categories, dropout_keep_prob):
    """VGG-like conv-net
    Args:
    training_batch: batch of images (N, 56, 56, 3)
    config: training configuration object
    Returns:
    class prediction scores
    """
    out = tf.cast(training_batch, tf.float32)
    out = (out - 128.0) / 128.0
    tf.summary.histogram('img', training_batch)

    # (N, 56, 56, 3)
    out = conv_2d_relu(out, 64, (3, 3), 'conv1_1')
    out = conv_2d_relu(out, 64, (3, 3), 'conv1_2')
    out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), name='pool1')

    # (N, 28, 28, 64)
    out = conv_2d_relu(out, 128, (3, 3), 'conv2_1')
    out = conv_2d_relu(out, 128, (3, 3), 'conv2_2')
    out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), name='pool2')

    # (N, 14, 14, 128)
    out = conv_2d_relu(out, 256, (3, 3), 'conv3_1')
    out = conv_2d_relu(out, 256, (3, 3), 'conv3_2')
    out = conv_2d_relu(out, 256, (3, 3), 'conv3_3')
    out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), name='pool3')

    # (N, 7, 7, 256)
    out = conv_2d_relu(out, 512, (3, 3), 'conv4_1')
    out = conv_2d_relu(out, 512, (3, 3), 'conv4_2')
    out = conv_2d_relu(out, 512, (3, 3), 'conv4_3')

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


def vgg_16_layer(training_batch, categories, dropout_prob, mode):
    out = tf.cast(training_batch, tf.float32)
    out = (out - 128.0) / 128.0
    tf.summary.histogram('img', training_batch)

    # (N, 56, 56, 3)
    out = tf.layers.conv2d(inputs=out, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    out = tf.layers.batch_normalization(out, training=mode)
    out = tf.layers.conv2d(inputs=out, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    out = tf.layers.batch_normalization(out, training=mode)
    out = tf.layers.max_pooling2d(inputs=out, pool_size=[2, 2], strides=2, name='pool1')

    # (N, 28, 28, 64)
    out = tf.layers.conv2d(inputs=out, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    out = tf.layers.conv2d(inputs=out, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    out = tf.layers.max_pooling2d(inputs=out, pool_size=[2, 2], strides=2, name='pool1')

    # (N, 14, 14, 128)
    out = tf.layers.conv2d(inputs=out, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    out = tf.layers.conv2d(inputs=out, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    out = tf.layers.conv2d(inputs=out, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    out = tf.layers.max_pooling2d(inputs=out, pool_size=[2, 2], strides=2, name='pool1')

    # (N, 7, 7, 256)
    out = tf.layers.conv2d(inputs=out, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    out = tf.layers.conv2d(inputs=out, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    out = tf.layers.conv2d(inputs=out, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

    # fc1: flatten -> fully connected layer
    out = tf.contrib.layers.flatten(out)

    # (N, 7, 7, 512) -> (N, 25088) -> (N, 4096)
    out = tf.layers.dense(inputs=out, units=4096, activation=tf.nn.relu)
    out = tf.layers.dropout(inputs=out, rate=dropout_prob, training=mode)

    # fc2
    # (N, 4096) -> (N, 2048)
    out = tf.layers.dense(inputs=out, units=2048, activation=tf.nn.relu)
    out = tf.layers.dropout(inputs=out, rate=dropout_prob, training=mode)

    # softmax
    # (N, 2048) -> (N, 200)
    logits = tf.layers.dense(inputs=out, units=categories, activation=tf.identity)

    return logits


if __name__ == "__main__":
    x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])

    logits = vgg_16(x, 200, 1.0)

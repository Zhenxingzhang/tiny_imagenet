import tensorflow as tf
import src.models.model as model
from tensorflow.examples.tutorials.mnist import input_data


def mnist_layer(training_batch, categories, dropout_keep_prob, mode):
    """VGG-like conv-net
    Args:
    training_batch: batch of images (N, 56, 56, 3)
    config: training configuration object
    Returns:
    class prediction scores
    """
    out = tf.cast(training_batch, tf.float32)
    out = tf.reshape(out, [-1, 28, 28, 1])

    # (N, 28, 28, 3)
    out = model.conv_2d_relu(out, 32, [5, 5], 'conv1')
    out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), name='pool1')

    # (N, 14, 14, 32)
    out = model.conv_2d_relu(out, 64, [5, 5], 'conv2')
    out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), name='pool2')

    # fc1: flatten -> fully connected layer
    # (N, 7, 7, 64) -> (N, 3136) -> (N, 4096)
    out = tf.contrib.layers.flatten(out)
    out = model.dense_relu(out, 1024, 'fc1')
    out = tf.layers.dropout(out, dropout_keep_prob, training=mode)

    # (N, 2048) -> (N, 200)
    logits = model.dense(out, categories, 'fc2')

    return logits


def mnist_basic(training_batch, categories, dropout_keep_prob):
    out = tf.cast(training_batch, tf.float32)
    out = tf.reshape(out, [-1, 28, 28, 1])
    out = model.conv_pool_layer(out, filter_size=5, num_filters=32, layer_name='conv_pool_1')
    out = model.conv_pool_layer(out, filter_size=5, num_filters=64, layer_name='conv_pool_2')
    out = model.fc_layer(out, num_units=7 * 7 * 64, layer_name='FC_1', keep_prob_tensor=dropout_keep_prob)
    logits = model.fc_layer(out, num_units=categories, layer_name='softmax', keep_prob_tensor=1.0, act=tf.identity)
    return logits


def cnn_model_fn(features, categories, dropout_prob, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features, [-1, 28, 28, 1])
    input_layer = tf.cast(input_layer, tf.float32)

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(inputs=dense, rate=dropout_prob, training=mode)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=categories)
    return logits


if __name__ == "__main__":
    x_input = tf.placeholder(tf.int8, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    keep_prob = tf.placeholder(tf.float32)

    logits = mnist_layer(x_input, 10, keep_prob, True)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=logits)
        cross_entropy = tf.reduce_mean(cross_entropy)
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
        #                                                         logits=logits)
        # cross_entropy = tf.reduce_mean(cross_entropy)

        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    learning_rate = 0.001

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    print(logits.shape)

    mnist = input_data.read_data_sets("/data/MNIST_data/", one_hot=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(1001):
            batch_images, batch_labels = mnist.train.next_batch(64)
            loss, _ = sess.run([cross_entropy, train_step],
                               feed_dict={x_input: batch_images, y_: batch_labels, keep_prob: 0.0})
            print("steps: {}, loss: {}".format(i, loss))


# """Convolutional Neural Network Estimator for MNIST, built with tf.layers."""
#
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import numpy as np
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
#
# tf.logging.set_verbosity(tf.logging.INFO)
#
#
# def cnn_model_fn(features, labels, mode):
#     """Model function for CNN."""
#     # Input Layer
#     # Reshape X to 4-D tensor: [batch_size, width, height, channels]
#     # MNIST images are 28x28 pixels, and have one color channel
#     input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
#
#     # Convolutional Layer #1
#     # Computes 32 features using a 5x5 filter with ReLU activation.
#     # Padding is added to preserve width and height.
#     # Input Tensor Shape: [batch_size, 28, 28, 1]
#     # Output Tensor Shape: [batch_size, 28, 28, 32]
#     conv1 = tf.layers.conv2d(
#       inputs=input_layer,
#       filters=32,
#       kernel_size=[5, 5],
#       padding="same",
#       activation=tf.nn.relu)
#
#     # Pooling Layer #1
#     # First max pooling layer with a 2x2 filter and stride of 2
#     # Input Tensor Shape: [batch_size, 28, 28, 32]
#     # Output Tensor Shape: [batch_size, 14, 14, 32]
#     pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
#
#     # Convolutional Layer #2
#     # Computes 64 features using a 5x5 filter.
#     # Padding is added to preserve width and height.
#     # Input Tensor Shape: [batch_size, 14, 14, 32]
#     # Output Tensor Shape: [batch_size, 14, 14, 64]
#     conv2 = tf.layers.conv2d(
#       inputs=pool1,
#       filters=64,
#       kernel_size=[5, 5],
#       padding="same",
#       activation=tf.nn.relu)
#
#     # Pooling Layer #2
#     # Second max pooling layer with a 2x2 filter and stride of 2
#     # Input Tensor Shape: [batch_size, 14, 14, 64]
#     # Output Tensor Shape: [batch_size, 7, 7, 64]
#     pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
#
#     # Flatten tensor into a batch of vectors
#     # Input Tensor Shape: [batch_size, 7, 7, 64]
#     # Output Tensor Shape: [batch_size, 7 * 7 * 64]
#     pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
#
#     # Dense Layer
#     # Densely connected layer with 1024 neurons
#     # Input Tensor Shape: [batch_size, 7 * 7 * 64]
#     # Output Tensor Shape: [batch_size, 1024]
#     dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
#
#     # Add dropout operation; 0.6 probability that element will be kept
#     dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
#
#     # Logits layer
#     # Input Tensor Shape: [batch_size, 1024]
#     # Output Tensor Shape: [batch_size, 10]
#     logits = tf.layers.dense(inputs=dropout, units=10)
#
#     predictions = {
#       # Generate predictions (for PREDICT and EVAL mode)
#       "classes": tf.argmax(input=logits, axis=1),
#       # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
#       # `logging_hook`.
#       "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
#     }
#
#     if mode == tf.estimator.ModeKeys.PREDICT:
#         return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
#
#     # Calculate Loss (for both TRAIN and EVAL modes)
#     # onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
#     loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
#
#     # Configure the Training Op (for TRAIN mode)
#     if mode == tf.estimator.ModeKeys.TRAIN:
#         optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
#         train_op = optimizer.minimize(
#             loss=loss,
#             global_step=tf.train.get_global_step())
#         return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
#
#     # Add evaluation metrics (for EVAL mode)
#     eval_metric_ops = {
#       "accuracy": tf.metrics.accuracy(
#           labels=labels, predictions=predictions["classes"])}
#
#     return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
#
#
# def main(unused_argv):
#     # Load training and eval data
#     mnist = input_data.read_data_sets("/data/MNIST_data/", one_hot=True)
#     train_data = mnist.train.images  # Returns np.array
#     train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#     eval_data = mnist.test.images  # Returns np.array
#     eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
#
#     # Create the Estimator
#     mnist_classifier = tf.estimator.Estimator(
#       model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
#
#     # Set up logging for predictions
#     # Log the values in the "Softmax" tensor with label "probabilities"
#     tensors_to_log = {"probabilities": "softmax_tensor"}
#     logging_hook = tf.train.LoggingTensorHook(
#       tensors=tensors_to_log, every_n_iter=50)
#
#     # Train the model
#     train_input_fn = tf.estimator.inputs.numpy_input_fn(
#       x={"x": train_data},
#       y=train_labels,
#       batch_size=100,
#       num_epochs=None,
#       shuffle=True)
#     mnist_classifier.train(
#       input_fn=train_input_fn,
#       steps=20000,
#       hooks=[logging_hook])
#
#     # Evaluate the model and print results
#     eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#       x={"x": eval_data},
#       y=eval_labels,
#       num_epochs=1,
#       shuffle=False)
#     eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
#     print(eval_results)
#
#
# if __name__ == "__main__":
#     tf.app.run()

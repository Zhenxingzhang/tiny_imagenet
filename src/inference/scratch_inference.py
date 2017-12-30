import tensorflow as tf
import numpy as np
from conv_mnist_tensorboard import conv_pool_layer, FC_layer
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

if __name__ == "__main__":
    # Create some variables.
    mnist = read_data_sets('data/mnist_data', one_hot=True)

    test_labels = mnist.test.labels
    test_images = mnist.test.images
    print(test_images.shape)

    # num_val = mnist.test.labels.shape[0]
    # np.random.seed(1)
    # eval_sample = np.random.choice(range(num_val), 10, replace=False, )
    # eval_label = np.apply_along_axis(lambda a: [i for i, j in enumerate(a) if j == max(a)][0], 1, y_labels[eval_sample])
    # eval_images = mnist.test.images[eval_sample]

    x = tf.placeholder(tf.float32, shape=[None, 28*28])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    with tf.name_scope('input_image'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])  # reshape x as a 4 tensor
        # x_image = tf.image.resize_images(x_image, [28, 28])

    out_1 = conv_pool_layer(x_image, filter_size=5, patches_in=1, num_filters=32, layer_name='conv_pool_1')
    out_2 = conv_pool_layer(out_1, filter_size=5, patches_in=32, num_filters=64, layer_name='conv_pool_2')
    out_3 = FC_layer(out_2, input_dim=7 * 7 * 64, output_dim=1024, layer_name='FC_1')
    # add dropout in this layer
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)  # use keep_prob as a placeholder so can modify later
        out_3_drop = tf.nn.dropout(out_3, keep_prob)

    # NOTE: here we do not apply the softmax activation directly, instead we apply just the identity
    # this is because we use the tensorflow function 'softmax_cross_entropy_with_logits' below which takes in
    # logits (the outputs before the softmax) and computes the cross entropy from them
    # the reason for this is that the softmax function ~ exp(z)/sum(exp(z)) can be numerically unstable becasue of the
    # exponents but since the cross entropy computes the log of this we can formulate in a more stable way
    y = FC_layer(out_3_drop, input_dim=1024, output_dim=10, layer_name='softmax', act=tf.identity)
    pred_label = tf.argmax(y, 1)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # initializer = tf.global_variables_initializer()
    # Add ops to save and restore all the variables.
    variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for i in variables_to_restore[:-2]:
        print(i.name)

    variables_to_initial = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="softmax")
    for i in variables_to_initial:
        print(i.name)

    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
        # sess.run(initializer)

        saver.restore(sess, "/data/checkpoints/conv_mnist/model.ckpt")

        # sess.run(tf.variables_initializer(variables_to_initial))

        # for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="softmax/biases"):
        #     print i.name
        #     print i.eval()
        # exit()

        predictions, scores, acc = sess.run([pred_label, y, accuracy], {x: test_images, y_: test_labels, keep_prob: 1.0})
        # print(test_labels)
        # print(scores)
        print(predictions)
        print(acc)


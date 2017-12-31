import tensorflow as tf
from src.models.model import conv_net
from src.common.paths import DATA_PATH, LOG_DIR, CHECKPOINT_PATH
from src.data_preparation import dataset
import os


if __name__ == "__main__":
    BATCH_SIZE = 256
    NUM_STEPS = 10001
    LEARNING_RATE = 1e-3

    with tf.name_scope("input"):
        input_images = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
        label = tf.placeholder(tf.int64)
        input_images_summary = tf.summary.image('input/images', input_images)

    with tf.name_scope('dropout_keep_prob'):
        keep_prob_tensor = tf.placeholder(tf.float32)

    logits = conv_net(input_images, 200, keep_prob_tensor)

    print(logits.shape)
    # for monitoring
    with tf.name_scope('loss'):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
        loss_mean = tf.reduce_mean(loss)
        tf.summary.scalar('loss', loss_mean)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), label)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_mean)

    summary_op = tf.summary.merge_all()

    variables_to_store = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    saver = tf.train.Saver(var_list=variables_to_store, max_to_keep=3)

    train_tfrecord_file = os.path.join(DATA_PATH, "train.tfrecord")
    val_tfrecord_file = os.path.join(DATA_PATH, "val.tfrecord")

    with tf.Session() as sess:
        next_train_batch = dataset.get_train_val_data_iter(sess, [train_tfrecord_file], batch_size=BATCH_SIZE)
        next_val_batch = dataset.get_train_val_data_iter(sess, [val_tfrecord_file], batch_size=BATCH_SIZE)

        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, str(LEARNING_RATE), "train"), sess.graph)
        val_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, str(LEARNING_RATE), "val"), sess.graph)

        print("Start training with ")

        for i in range(NUM_STEPS):
            train_batch_examples = sess.run(next_train_batch)
            train_images = train_batch_examples["image_shape"]
            train_labels = train_batch_examples["label"]

            _, step_loss, step_summary = sess.run([train_op, loss_mean, summary_op],
                                                  feed_dict={input_images: train_images,
                                                             label: train_labels,
                                                             keep_prob_tensor: 1.0})
            train_writer.add_summary(step_summary, i)
            print("Step {}, train loss: {}".format(i, step_loss))

            if i % 10 == 0:
                saver.save(sess, os.path.join(CHECKPOINT_PATH, str(LEARNING_RATE), "model.ckpt"))

                val_batch_examples = sess.run(next_val_batch)
                val_images = val_batch_examples["image_shape"]
                val_labels = val_batch_examples["label"]

                val_step_loss, val_step_summary = sess.run([loss_mean, summary_op],
                                                           feed_dict={input_images: val_images,
                                                                      label: val_labels,
                                                                      keep_prob_tensor: 1.0})
                val_writer.add_summary(val_step_summary, i)
            #     print("Step {}, val loss: {}".format(i, val_step_loss))

    print("Finish training from scratch...")

import tensorflow as tf
from src.models.model import conv_net
from src.data_preparation.dataset import read_record_to_queue
from src.common.paths import DATA_PATH, LOG_DIR, CHECKPOINT_PATH
import os


def prepocess_ims(image):
    return tf.cast(image, tf.float32) / 255.


if __name__ == "__main__":
    BATCH_SIZE = 128
    NUM_STEPS = 1001
    LEARNING_RATE = 1e-3

    with tf.name_scope("input"):
        input_images = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
        label = tf.placeholder(tf.int32)

    with tf.name_scope('dropout_keep_prob'):
        keep_prob_tensor = tf.placeholder(tf.float32)

    logits = conv_net(input_images, keep_prob_tensor)

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

    # variables_to_store = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # saver = tf.train.Saver(var_list=variables_to_store, max_to_keep=3)

    train_tfrecord_file = os.path.join(DATA_PATH, "train.tfrecord")
    # val_tfrecord_file = os.path.join(DATA_PATH, "val.tfrecord")

    with tf.name_scope('data'):
        shapes = {"image": (64, 64, 3), "label": 1}

        train_images_batch, train_labels_batch = read_record_to_queue(train_tfrecord_file,
                                                                      shapes,
                                                                      preproc_func=prepocess_ims,
                                                                      batch_size_=BATCH_SIZE)

        train_images_summary = tf.summary.image('train/images', train_images_batch)

        # val_images_batch, val_labels_batch = read_record_to_queue(val_tfrecord_file,
        #                                                           shapes,
        #                                                           preproc_func=prepocess_ims,
        #                                                           batch_size_=BATCH_SIZE)
        # val_images_summary = tf.summary.image('eval/images', val_images_batch[:10])

    train_merged = tf.summary.merge([train_images_summary])
    # val_merged = tf.summary.merge([val_images_summary])

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()

        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, str(LEARNING_RATE), "train"), sess.graph)
        # val_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, str(LEARNING_RATE), "val"), sess.graph)

        print("Start training with ")
        sess.run(init)
        tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(NUM_STEPS):
            train_images, train_labels, train_data_summary = \
                sess.run([train_images_batch, train_labels_batch, train_merged])
            train_writer.add_summary(train_data_summary, i)

            _, step_loss, step_summary = sess.run([train_op, loss_mean, summary_op],
                                                  feed_dict={input_images: train_images,
                                                             label: train_labels,
                                                             keep_prob_tensor: 0.5})
            train_writer.add_summary(step_summary, i)
            print("Step {}, train loss: {}".format(i, step_loss))

            # if i % 10 == 0:
            #     saver.save(sess, os.path.join(CHECKPOINT_PATH, str(LEARNING_RATE), "model.ckpt"))
            #
            #     val_images, val_labels, val_summary = sess.run([val_images_batch, val_labels_batch, val_merged])
            #     val_writer.add_summary(val_summary, i)
            #
            #     val_step_loss, val_step_summary = sess.run([loss_mean, summary_op],
            #                                                feed_dict={input_images: val_images,
            #                                                           label: val_labels,
            #                                                           keep_prob_tensor: 1.0})
            #     val_writer.add_summary(val_step_summary, i)
            #     print("Step {}, val loss: {}".format(i, val_step_loss))

        coord.request_stop()

    print("Finish training from scratch...")

import tensorflow as tf
from src.models.model import conv_net
from src.data_preparation.dataset import read_record_to_queue
from src.common.paths import DATA_PATH, LOG_DIR
import os

def prepocess_ims(image):
    return tf.cast(image, tf.float32) / 255.


if __name__ == "__main__":
    BATCH_SIZE = 128
    NUM_STEPS = 100
    LEARNING_RATE = 1e-5

    train_tfrecord_file = os.path.join(DATA_PATH, "train.tfrecord")
    val_tfrecord_file = os.path.join(DATA_PATH, "val.tfrecord")

    with tf.name_scope('data'):
        shapes = {"image": (64, 64, 3), "label": 1}

        train_images_batch, train_labels_batch = read_record_to_queue(train_tfrecord_file,
                                                                      shapes,
                                                                      preproc_func=prepocess_ims,
                                                                      batch_size_=BATCH_SIZE)
        # Display the training images in the visualizer.
        train_images_summary = tf.summary.image('train/images', train_images_batch[:10])

        val_images_batch, val_labels_batch = read_record_to_queue(val_tfrecord_file,
                                                                  shapes,
                                                                  preproc_func=prepocess_ims,
                                                                  batch_size_=BATCH_SIZE)
        # Display the training images in the visualizer.
        val_images_summary = tf.summary.image('eval/images', val_images_batch[:10])

    with tf.name_scope("input"):
        input_images = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])

    logits = conv_net(input_images, keep_prob_=.5)

    train_merged = tf.summary.merge([train_images_summary])
    val_merged = tf.summary.merge([val_images_summary])

    with tf.Session() as sess:

        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, str(LEARNING_RATE), "train"), sess.graph)
        val_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, str(LEARNING_RATE), "val"), sess.graph)

        print("Start training with ")
        tf.train.start_queue_runners(sess=sess)
        for i in range(NUM_STEPS):
            images, summary = sess.run([train_images_batch, train_merged])
            print(images[0][0][0])
            train_writer.add_summary(summary, i)
            print(images.shape)

            val_images, val_summary = sess.run([val_images_batch, val_merged])
            print(val_images[0][0][0])
            val_writer.add_summary(val_summary, i)
            print(val_images.shape)

    print("Finish training from scratch...")

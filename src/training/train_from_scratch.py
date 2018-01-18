import tensorflow as tf
from src.common import paths
from src.data_preparation import dataset
from src.models import raw_model, raw_model
import yaml
import os
import datetime
import argparse


def train(model_name, model_arch, train_bz, val_bz, keep_prob_rate, steps, l_rate, input_h, input_w, categories):

    dataset.IMAGE_HEIGHT = input_h
    dataset.IMAGE_WIDTH = input_w

    with tf.name_scope("input"):
        input_images = tf.placeholder(tf.float32, shape=[None, input_h, input_w, 3])
        label = tf.placeholder(tf.int64)
        tf.summary.image('images', input_images)

    with tf.name_scope('dropout_keep_prob'):
        keep_prob_tensor = tf.placeholder(tf.float32)

    if model_arch == "mnist_net":
        logits = raw_model.mnist_net(input_images, categories, keep_prob_tensor)
    elif model_arch == "conv_net":
        logits = raw_model.conv_net_1(input_images, categories, keep_prob_tensor)
    elif model_arch == "vgg_16":
        logits = raw_model.vgg_16(input_images, categories, keep_prob_tensor)
    else:
        print("Model arch error, {} does not exist".format(model_arch))
        exit()

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

    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(l_rate, global_step,
                                               3000, 0.5, staircase=True)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_mean, global_step=global_step)

    summary_op = tf.summary.merge_all()

    variables_to_store = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    saver = tf.train.Saver(var_list=variables_to_store)

    train_tfrecord_file = paths.TRAIN_TF_RECORDS
    val_tfrecord_file = paths.VAL_TF_RECORDS

    with tf.Session() as sess:
        next_train_batch = dataset.get_data_iter(sess, [train_tfrecord_file], "train", batch_size=train_bz)
        next_val_batch = dataset.get_data_iter(sess, [val_tfrecord_file], "val", batch_size=val_bz)

        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(
            os.path.join(paths.TRAIN_SUMMARY_DIR,
                         model_arch,
                         str(l_rate),
                         datetime.datetime.now().strftime("%Y%m%d-%H%M")),
            sess.graph)
        val_writer = tf.summary.FileWriter(
            os.path.join(paths.VAL_SUMMARY_DIR,
                         model_arch,
                         str(l_rate),
                         datetime.datetime.now().strftime("%Y%m%d-%H%M")),
            sess.graph)

        print("Start training with ")

        checkpoint_dir = os.path.join(paths.CHECKPOINT_DIR, model_name, str(l_rate))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        for i in range(steps):
            train_batch_examples = sess.run(next_train_batch)
            train_images = train_batch_examples["image_resize"]
            train_labels = train_batch_examples["label"]

            lr, _, step_loss, step_summary = sess.run([learning_rate, train_op, loss_mean, summary_op],
                                                      feed_dict={input_images: train_images,
                                                                 label: train_labels,
                                                                 keep_prob_tensor: keep_prob_rate})
            train_writer.add_summary(step_summary, i)
            print("Step {}, lr:{},  train loss: {}".format(i, lr, step_loss))

            if i % 100 == 0:
                saver.save(sess, os.path.join(checkpoint_dir, "model.ckpt"))

                val_batch_examples = sess.run(next_val_batch)
                val_images = val_batch_examples["image_resize"]
                val_labels = val_batch_examples["label"]

                val_step_loss, val_step_summary = sess.run([loss_mean, summary_op],
                                                           feed_dict={input_images: val_images,
                                                                      label: val_labels,
                                                                      keep_prob_tensor: 1.0})
                val_writer.add_summary(val_step_summary, i)
            #     print("Step {}, val loss: {}".format(i, val_step_loss))

    print("Finish training from scratch...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Default argument')
    parser.add_argument('-c',
                        dest="config_filename", type=str, required=True,
                        help='the config file name must be provide')
    args = parser.parse_args()

    with open(args.config_filename, 'r') as yml_file:
        cfg = yaml.load(yml_file)

    MODEL_NAME = str(cfg["MODEL"]["MODEL_NAME"])
    MODEL_ARCH = str(cfg["MODEL"]["MODEL_ARCH"])
    INPUT_HEIGHT = int(cfg["MODEL"]["INPUT_HEIGHT"])
    INPUT_WIDTH = int(cfg["MODEL"]["INPUT_WIDTH"])
    CATEGORIES = int(cfg["MODEL"]["CLASSES"])

    TRAIN_BATCH_SIZE = int(cfg["TRAIN"]["BATCH_SIZE"])
    TRAIN_STEPS_COUNT = int(cfg["TRAIN"]["EPOCHS_COUNT"])
    TRAIN_LEARNING_RATE = float(cfg["TRAIN"]["LEARNING_RATE"])
    TRAIN_KEEP_PROB = float(cfg['TRAIN']['KEEP_PROB'])
    TRAIN_TF_RECORDS = str(cfg["TRAIN"]["TF_RECORDS"])

    EVAL_BATCH_SIZE = cfg["TRAIN"]["BATCH_SIZE"]
    EVAL_TF_RECORDS = str(cfg["TRAIN"]["TF_RECORDS"])

    train(MODEL_NAME, MODEL_ARCH, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, TRAIN_KEEP_PROB, TRAIN_STEPS_COUNT,
          TRAIN_LEARNING_RATE, INPUT_HEIGHT, INPUT_WIDTH, CATEGORIES)



import tensorflow as tf
from src.models.model import conv_net
from src.common import paths
from src.data_preparation import dataset
import os


if __name__ == "__main__":

    LEARNING_RATE = 1e-3

    _, label_decoder = dataset.get_text_labels()

    input_images = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
    keep_prob_tensor = tf.placeholder(tf.float32)

    logits = conv_net(input_images, 200, keep_prob_tensor)
    prediction = tf.argmax(logits, 1)
    # initializer = tf.global_variables_initializer()
    # Add ops to save and restore all the variables.
    variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    saver = tf.train.Saver(variables_to_restore)

    check_point_path = os.path.join(paths.CHECKPOINT_PATH, str(LEARNING_RATE), "model.ckpt")

    with tf.Session() as sess:

        print("Restore from {}".format(check_point_path))
        saver.restore(sess, check_point_path)

        test_tfrecord_file = os.path.join(paths.DATA_PATH, "test.tfrecord")

        next_test_batch = dataset.get_test_data_iter(sess, [test_tfrecord_file])

        with open(os.path.join(paths.OUTPUT_PATH, "prediction.txt"), "w") as output:
            try:
                while True:
                    test_batch_examples = sess.run(next_test_batch)
                    test_images = test_batch_examples["image_shape"]
                    test_filename = test_batch_examples["filename"]

                    pred_labels = sess.run(prediction, {input_images: test_images, keep_prob_tensor: 1.0})

                    for (p_label, filename) in zip(pred_labels, test_filename):
                        output.writelines("{} {}\n".format(filename, label_decoder[p_label]))

            except tf.errors.OutOfRangeError:
                print('End of the dataset')

    print("Finish...")

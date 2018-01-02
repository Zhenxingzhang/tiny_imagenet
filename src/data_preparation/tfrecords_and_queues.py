"""
Up to now we have held all data in memory. This is of course impossible with large datasets.

In this file we explore the use of TFRecords (binary files quickly loading data from disk) and Queues to store
asynchronously loading data.

In this example we the TinyImageNet-200 dataset which has 100,000 64x64 images for 200 classes

We will examine 2 options for reading from TFRecord files:
a) reading from the record directly one example at a time
b) reading from the record into a queue and sampling batches from that queue

For more info, consult the great documentation on this from Tensorflow at
https://www.tensorflow.org/versions/r0.12/how_tos/reading_data/index.html
"""
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib

# to remove issue with default matplotlib backend (causing runtime error "python is not installed as a framework")
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os
from src.common.paths import DATA_PATH


def grey_to_rgb(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret


def csv_to_record(csv_file, tfrecord_file):
    with open(csv_file) as f:
        lines = f.readlines()
        np.random.shuffle(lines)

    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    # iterate over each example
    # wrap with tqdm for a progress bar

    for line in tqdm(lines):
        path = line.split(',')[0]
        image = np.array(Image.open(path))
        image_name = path.split("/")[-1]
        if len(image.shape) == 2:
            # there are some greyscale image in data, reformat them
            image = grey_to_rgb(image)

        flat_image = image.flatten().astype("int64")
        text_label = line.split(',')[1]
        label = -1 if (text_label == '' or text_label is None) else int(text_label)

        # construct the Example proto object
        example = tf.train.Example(
            # Example contains a Features proto object
            features=tf.train.Features(
                # Features contains a map of string to Feature proto objects
                # A Feature contains one of either a int64_list,
                # float_list, or bytes_list
                feature={'label': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[label])),
                    'image': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=flat_image)),
                    'filename': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[image_name]))
                }
            )
        )

        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        writer.write(serialized)


def read_record_to_queue(tf_record_name, shapes, plot=None):
    def read_and_decode_single_example(filename):
        # first construct a queue containing a list of filenames.
        # this lets a user split up there dataset in multiple files to keep
        # size down
        filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
        # Unlike the TFRecordWriter, the TFRecordReader is symbolic
        # this means it creates a function and adds it to our graph that can be evaluation with our session
        # Each time we evaluate it it will pull the next batch off the queue and return that data
        reader = tf.TFRecordReader()
        # One can read a single serialized example from a filename
        # serialized_example is a Tensor of type string.
        _, serialized_example = reader.read(filename_queue)
        # The serialized example is converted back to actual values.
        # One needs to describe the format of the objects to be returned
        features = tf.parse_single_example(
            serialized_example,
            features={
                # We know the length of both fields. If not the
                # tf.VarLenFeature could be used
                'label': tf.FixedLenFeature([shapes['label']], tf.int64),
                'image': tf.FixedLenFeature([np.product(shapes['image'])], tf.int64)
            })
        # now return the converted data
        label = features['label']
        image = features['image']
        return label, image

    # returns symbolic label and image
    label, image = read_and_decode_single_example(tf_record_name)

    # groups examples into batches randomly
    # min_after_queue = size of buffer that will be randomly sampled
    # capcity = maxmimum examples to prefetch
    images_batch, labels_batch = tf.train.shuffle_batch([image, label], batch_size=32,
                                                        capacity=2000, min_after_dequeue=1000)

    sess = tf.Session()

    # Initialize graph
    sess.run(tf.global_variables_initializer())

    tf.train.start_queue_runners(sess=sess)

    # grab examples back.
    print('Reading random batches of 32')
    if plot:
        plt.suptitle('Read in batches from queue')
        for i in range(plot):
            # get ith batch
            label_vals, image_vals = sess.run([labels_batch, images_batch])
            idx = np.random.randint(0, 32)  # sample 1 instance from batch
            label_val = np.array(label_vals)[idx]
            if np.array(label).size > 1:
                label_val = np.argmax(label_val)
            image_val = np.array(image_vals)[idx]
            plt.subplot(3, plot / 3 + (1 if plot % 3 > 0 else 0), i + 1)
            plt.xticks(())
            plt.yticks(())
            plt.title(label_val)
            plt.imshow(image_val.reshape(shapes['image']).astype("uint8"))
        plt.show()
    else:
        for i in range(5):
            label_vals, image_vals = sess.run([labels_batch, images_batch])
            print('Labels of batch {} : {}'.format(i, label_vals))
            if i == 10:
                print("That's enough of that!")
                break


if __name__ == '__main__':

    # create TFRecords from csv files if necessary
    for set_name in ['train', 'val', 'test']:
        tfrecord_path = os.path.join(DATA_PATH, "{}.tfrecord".format(set_name))
        if not os.path.exists(tfrecord_path):
            print('Creating TFRecord from csv files for set: {}'.format(set_name))
            train_csv = os.path.join(DATA_PATH, "{}.csv".format(set_name))
            csv_to_record(train_csv, tfrecord_path)
        else:
            print('TFRecord for {} exists, nothing to do'.format(set_name))

    PLOT = 10  # number of images to plot (set == None to suppress plotting)

    # read from record one at time
    print('Reading from record one at a time')
    val_tfrecord_file = os.path.join(DATA_PATH, "train.tfrecord")
    # read_from_record(val_tfrecord_file, shapes={'label': 1, 'image': (64, 64, 3)},
    #                  plot=PLOT)

    # read from record into queue, shuffle and batch
    print('Reading from record into queue, random sample from queue in batches')
    read_record_to_queue(val_tfrecord_file, shapes={'label': 1, 'image': (64, 64, 3)},
                         plot=PLOT)

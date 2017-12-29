import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from src.common.paths import DATA_PATH
import os


def read_from_record(filename, shapes, n=10):
    records = []
    # create iterator from TFRecord and read examples from it
    count = 0
    for i, serialized_example in enumerate(tf.python_io.tf_record_iterator(filename)):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)  # deserialise

        record = dict()
        # traverse the Example format to get data
        record["image"] = np.array(example.features.feature['image'].int64_list.value).reshape(shapes['image'])
        record["label"] = example.features.feature['label'].int64_list.value[0]
        record["filename"] = example.features.feature['filename'].bytes_list.value[0]
        records.append(record)
        count +=1
        if count >= n:
            break
    return records


def read_record_to_queue(tf_record_name, shapes, preproc_func=None, batch_size_=32):
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
        label__ = features['label']
        image__ = tf.reshape(features['image'], [64, 64, 3])
        preproc_image = preproc_func(image__) if preproc_func is not None else image__

        return preproc_image, label__

    # returns symbolic label and image
    image_, label_ = read_and_decode_single_example(tf_record_name)

    # groups examples into batches randomly
    # min_after_queue = size of buffer that will be randomly sampled
    # capcity = maxmimum examples to prefetch
    images_batch_, labels_batch_ = tf.train.shuffle_batch([image_, label_],
                                                          batch_size=batch_size_,
                                                          capacity=2000,
                                                          min_after_dequeue=1000)
    return  images_batch_, labels_batch_


if __name__ == "__main__":
    val_tfrecord_file = os.path.join(DATA_PATH, "train.tfrecord")

    sample_count = 2
    samples = read_from_record(val_tfrecord_file, shapes={'label': 1, 'image': (64, 64, 3)}, n=sample_count)
    print(len(samples))
    for i in range(sample_count):
        print(samples[i]["label"])
        image = samples[i]["image"]
        plt.imshow(image.astype("uint8"))
        plt.show()

    images_batch, labels_batch = read_record_to_queue(val_tfrecord_file, shapes={'label': 1, 'image': (64, 64, 3)})

    with tf.Session() as sess:

        tf.train.start_queue_runners(sess=sess)

        # grab examples back.
        print('Reading random batches of 32')

        # get ith batch
        image_vals, label_vals = sess.run([labels_batch, images_batch])
        print(label_vals.shape)
        print(image_vals.shape)

        idx = np.random.randint(0, 32)  # sample 1 instance from batch
        label_val = label_vals[idx]
        print(label_val)

        image_val = image_vals[idx]

        plt.imshow(image_val.astype("uint8"))
        plt.show()

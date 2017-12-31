import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from src.common.paths import DATA_PATH, DATA_DIR
import os


def get_text_labels():
    labels_ = [lab for lab in os.listdir(os.path.join(DATA_DIR, 'train')) if lab[0] != '.']
    encoder_ = {lab: i for i, lab in enumerate(labels_)}
    decoder_ = {i: lab for i, lab in enumerate(labels_)}

    return encoder_, decoder_


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
        count += 1
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
                'image': tf.FixedLenFeature([np.product(shapes['image'])], tf.int64),
                'filename': tf.FixedLenFeature([], tf.string)
            })
        # now return the converted data
        label__ = tf.squeeze(features['label'])
        image__ = tf.reshape(features['image'], [64, 64, 3])
        filename__ = features['filename']
        preproc_image = preproc_func(image__) if preproc_func is not None else image__

        return preproc_image, label__, filename__

    # returns symbolic label and image
    image_, label_, filename_ = read_and_decode_single_example(tf_record_name)

    # groups examples into batches randomly
    # min_after_queue = size of buffer that will be randomly sampled
    # capcity = maxmimum examples to prefetch
    images_batch_, labels_batch_, filename_batch_ = tf.train.shuffle_batch([image_, label_, filename_],
                                                          batch_size=batch_size_,
                                                          capacity=20000,
                                                          min_after_dequeue=1000)
    return images_batch_, labels_batch_, filename_batch_


# dataset API
def read_train_image_record(record):
    _features = tf.parse_single_example(
        record,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([np.product((64, 64, 3))], tf.int64)
        })

    _features["image_shape"] = tf.reshape(tf.cast(_features['image'], tf.float32) / 255.0, [64, 64, 3])

    return _features


def image_dataset():
    _file_names = tf.placeholder(tf.string)
    _ds = tf.contrib.data.TFRecordDataset(_file_names).map(read_train_image_record)
    return _ds, _file_names


def get_train_val_data_iter(sess_, tf_records_paths_, buffer_size=20000, batch_size=64):
    ds_, file_names_ = image_dataset()
    ds_iter = ds_.shuffle(buffer_size).repeat().batch(batch_size).make_initializable_iterator()
    sess_.run(ds_iter.initializer, feed_dict={file_names_: tf_records_paths_})
    return ds_iter.get_next()


def read_test_image_record(record):
    feature_ = tf.parse_single_example(
        record,
        features={
            'image': tf.FixedLenFeature([np.product((64, 64, 3))], tf.int64),
            'filename': tf.FixedLenFeature([], tf.string)
        })

    feature_["image_shape"] = tf.reshape(tf.cast(feature_['image'], tf.float32) / 255.0, [64, 64, 3])

    return feature_


def test_image_dataset():
    file_names_ = tf.placeholder(tf.string)
    ds_ = tf.contrib.data.TFRecordDataset(file_names_).map(read_test_image_record)

    return ds_, file_names_


def get_test_data_iter(sess_, tf_records_paths_, batch_size=64):
    ds_, file_names_ = test_image_dataset()
    ds_iter = ds_.batch(batch_size).make_initializable_iterator()
    sess_.run(ds_iter.initializer, feed_dict={file_names_: tf_records_paths_})
    return ds_iter.get_next()


if __name__ == "__main__":
    _, label_decoder = get_text_labels()
    val_tfrecord_file = os.path.join(DATA_PATH, "val.tfrecord")

    sample_count = 2
    samples = read_from_record(val_tfrecord_file, shapes={'label': 1, 'image': (64, 64, 3)}, n=sample_count)
    for i in range(sample_count):
        print(label_decoder[samples[i]["label"]])
        image = samples[i]["image"]
        plt.imshow(image.astype("uint8"))
        plt.show()

    images_batch, labels_batch, file_batch = read_record_to_queue(val_tfrecord_file, shapes={'label': 1, 'image': (64, 64, 3)})

    with tf.Session() as sess:
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #
        # # grab examples back.
        # print('Reading random batches of 32')
        #
        # # get ith batch
        # image_vals, label_vals, files_val = sess.run([images_batch, labels_batch, file_batch])
        # print(label_vals)
        # print(image_vals.shape)
        #
        # idx = np.random.randint(0, 32)  # sample 1 instance from batch
        # label_val = label_vals[idx]
        # print(label_decoder[label_val])
        # print(files_val[idx])
        #
        # image_val = image_vals[idx]
        #
        # plt.imshow(image_val.astype("uint8"))
        # plt.show()
        #
        # coord.request_stop()
        # coord.join(threads)

        test_tfrecord_file = os.path.join(DATA_PATH, "test.tfrecord")

        next_test_batch = get_test_data_iter(sess, [test_tfrecord_file])
        batch_examples = sess.run(next_test_batch)
        images = batch_examples["image_shape"]
        print(images.shape)
        # labels = batch_examples["label"]
        # print(labels.shape)
        # _, decoder = get_text_labels()
        # print(decoder[labels[0]])

        batch_filename = batch_examples["filename"]
        print(batch_filename.shape)
        print(batch_filename[0])

        plt.imshow(images[0])
        plt.show()


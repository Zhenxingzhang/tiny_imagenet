import os

DATA_NAME = "tiny_imagenet"
DATA_ROOT = "/data/tiny_imangenet/"
DATA_PATH = "/data/tiny_imangenet/tiny_imagenet_data"
DATA_DIR = "/data/tiny_imangenet/tiny-imagenet-200"
LOG_DIR = "/data/summary/tiny_imagenet"
CHECKPOINT_DIR = "/data/checkpoints/tiny_imagenet"
OUTPUT_PATH = "/data/outputs/tiny_imagenet"

TRAIN_TF_RECORDS = os.path.join(DATA_PATH, 'train.tfrecord')
VAL_TF_RECORDS = os.path.join(DATA_PATH, 'val.tfrecord')

TRAIN_SUMMARY_DIR = os.path.join(LOG_DIR, "train")
VAL_SUMMARY_DIR = os.path.join(LOG_DIR, "val")

TEST_TF_RECORDS = os.path.join(DATA_PATH, 'test.tfrecord')

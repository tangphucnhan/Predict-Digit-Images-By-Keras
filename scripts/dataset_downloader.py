import os
import shutil

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from constants import *


def download_tensorflow_datasets(name: str, data_dir: str):
    (data_train, data_test), data_info = tfds.load(
        name=name,
        data_dir=data_dir,
        with_info=True,
        as_supervised=True,
        shuffle_files=False,
        # split=[tfds.Split.TRAIN, tfds.Split.TEST]
        split=['train[:100%]', 'test[:100%]']
    )
    if os.path.exists(f"{data_dir}/downloads"):
        shutil.rmtree(f"{data_dir}/downloads", ignore_errors=True)

    return data_info


def load_and_split(tfds_name, version, data_dir, test_size=0.2, shuffle=False):
    if not tfds_name:
        print("Tensorflow dataset name is required")
        exit()
    if not data_dir:
        print("Provide directory to read/write data")
        exit()
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    version_path = f"{data_dir}/{tfds_name}/{version}"

    if not os.path.exists(version_path):
        download_tensorflow_datasets(tfds_name, data_dir)

    builder = tfds.builder_from_directory(version_path)
    if builder.version is None:
        print("Build dataset from data directory unsuccessfully")
        exit()

    dataset: tf.data.Dataset = builder.as_dataset(split='train[0:]')

    batch_size = 32
    dataset_batches = dataset.batch(batch_size)
    dataset_x = np.concatenate([d["image"] for d in dataset_batches], axis=0)
    dataset_y = np.concatenate([d["label"] for d in dataset_batches], axis=0)
    x_train, x_test, y_train, y_test = train_test_split(
        dataset_x,
        dataset_y,
        test_size=test_size,
        shuffle=shuffle,
    )
    return x_train, x_test, y_train, y_test

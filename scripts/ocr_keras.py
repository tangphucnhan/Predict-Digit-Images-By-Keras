import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from dataset_downloader import *
from constants import *


def one_hot(arr, k, dtype=np.float32):
    return np.array(arr[:, None] == np.arange(k), dtype)


force_create_new = len(sys.argv) == 2 and sys.argv[1] == "1"
LABELS_COUNT = 10

x_train, x_test, y_train, y_test = load_and_split("mnist", "3.0.1", data_dir=RESOURCES_PATH, test_size=0.2)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

model = None
if not force_create_new and os.path.exists(f"{MODELS_PATH}/ocr_keras.model"):
    print("...Loading saved model")
    model: tf.keras.models.Sequential = tf.keras.models.load_model(f"{MODELS_PATH}/ocr_keras.model")

if model is None:
    print("Creating new model")
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(LABELS_COUNT, activation="softmax"))
    # sigmoid result in lower accuracy than softmax
    # model.add(tf.keras.layers.Dense(LABELS_COUNT, activation="sigmoid"))

    # Result: 9/10, 10/10
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy.__name__,
                  optimizer=keras.optimizers.Adam.__name__,
                  metrics=['accuracy'])

    # Result: 9/10, 10/10
    # model.compile(loss=keras.losses.SparseCategoricalCrossentropy.__name__,
    #               optimizer=keras.optimizers.Nadam.__name__,
    #               metrics=['accuracy'])

    # Result: 2/10
    # model.compile(loss=keras.losses.SparseCategoricalCrossentropy.__name__,
    #               optimizer=keras.optimizers.SGD.__name__,
    #               metrics=['accuracy'])

    # Result: 8/10, 9/10
    # model.compile(loss=keras.losses.SparseCategoricalCrossentropy.__name__,
    #               optimizer=keras.optimizers.RMSprop.__name__,
    #               metrics=['accuracy'])

    # Result: 8/10, 9/10
    # model.compile(loss=keras.losses.CategoricalFocalCrossentropy.__name__,
    #               optimizer=keras.optimizers.RMSprop.__name__,
    #               metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=32)

    model.save(f"{MODELS_PATH}/ocr_keras.model")

loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
print("loss:", loss)
print(f"accuracy: {accuracy:.2f}")


# Testing
match_count = 0
for i in range(LABELS_COUNT):
    img = cv2.imread(f"{TEST_IMAGES_PATH}/numbers/{i}.{IMG_EXT}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("img", img)
    img = np.reshape(img, (1, img.shape[0], img.shape[1]))
    pred_cases = model.predict(img, verbose=False)
    # print("--------------")
    # print(f"i: {i}, guess:\n{pred_cases}")
    idx = np.argmax(pred_cases)
    print(f"i: {i}, max acc: {np.max(pred_cases):2.2f}, prediction: {idx}")
    if idx == i:
        match_count += 1
    # cv2.waitKey(0)
print(f"Matched: {match_count}/{LABELS_COUNT}")

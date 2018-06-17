from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt
from mnist_demo import *

import tensorflow as tf
import tensorflow.contrib.eager as tfe

import numpy as np
from keras.datasets import mnist

# download the mnist
# x:灰度图数据；y:标签
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


# data pre-processing
# 4-D tensor: [batch_size, width, height, channels]
X_train = X_train.reshape(-1, 28, 28, 1)/255.0
X_test = X_test.reshape(-1, 28, 28, 1)/255.0
# Converts a class vector (integers) to binary class matrix.
Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=10)
Y_test = tf.keras.utils.to_categorical(Y_test, num_classes=10)


# Build CNN
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(
    input_shape=(28, 28, 1),
    filters=32, kernel_size=[5, 5],
    padding='same', activation=tf.nn.relu))

model.add(tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2))

model.add(tf.keras.layers.Conv2D(
    filters=64, kernel_size=[5, 5],
    padding='same', activation=tf.nn.relu))

model.add(tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(units=1024, activation=tf.nn.relu))

model.add(tf.keras.layers.Dropout(rate=0.4))

model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

print(model.summary())

# Define optimizer
model.compile(loss='mean_squared_error', optimizer='sgd')

# Training
print('Training')
model.fit(X_train, Y_train, epochs=1, verbose=1, batch_size=128, validation_data=(X_test, Y_test))

# testing
score = model.evaluate(X_test, Y_test, verbose=0)
print(model.metrics_names)
print('Test loss:', score)



dir_name = 'C:\\TensorFlow_train\\identify'
files = os.listdir(dir_name)
cnt = len(files)
for i in range(cnt):
    files[i] = dir_name+"/"+files[i]
    test_images, test_labels = GetImage([files[i]])
    # predict
    predict_labels = model.predict(test_images)
    print("number:", test_labels, "\n", "prediction", predict_labels)



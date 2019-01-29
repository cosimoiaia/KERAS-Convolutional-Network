#!/usr/bin/env python

##########################################
#
# keras-convolution.py: A Simple implementation of a Convolutional Neural Network with Keras 
#                used to classify the MNIST Dataset of handwritten digits (Source: http://yann.lecun.com/exdb/mnist/).
#  
#
# Author: Cosimo Iaia <cosimo.iaia@gmail.com>
# Date: 10/10/2017
#
# This file is distribuited under the terms of GNU General Public
#
########################################


from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import argparse


FLAGS = None

def main():
    batch_size = FLAGS.batch_size
    num_classes = FLAGS.class_number
    epochs = FLAGS.epochs
    path = FLAGS.model_file

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs, 
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parametized Convolutional Neural Network in Keras")
    parser.add_argument('--batch_size', type=int, default='128', help='How many images train on at a time')
    parser.add_argument('--epochs', type=int, default='12', help='How many epochs to train')
    parser.add_argument('--model_file', type=str, default='model.tfl', help='Path to save the model file')
    parser.add_argument('--class_number', type=int, default=10, help='Number of classes to train on')
    FLAGS = parser.parse_args()
    main()
    
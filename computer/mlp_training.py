__author__ = 'zhengwang'

import cv2
import numpy as np
import glob
import tensorflow as tf
import keras.backend
from keras.layers import Flatten, Dense, Dropout, Lambda, Convolution2D, Cropping2D, Conv2D, Conv1D, Reshape
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import matplotlib.pyplot as plt

print 'Loading training data...'
e0 = cv2.getTickCount()

# load training data
image_array = np.zeros((1, 38400))
label_array = np.zeros((1, 4), 'float')
training_data = glob.glob('training_data/*.npz')

for single_npz in training_data:
    with np.load(single_npz) as data:
        print data.files
        train_temp = data['train']
        train_labels_temp = data['train_labels']
        print train_temp.shape
        print train_labels_temp.shape
    image_array = np.vstack((image_array, train_temp))
    label_array = np.vstack((label_array, train_labels_temp))

train = image_array[1:, :]
train_labels = label_array[1:, :]
print train.shape
print train_labels.shape

show_images = False
export_images = False
should_train = True
use_tensorflow = True


def show_image(sample, delay=200):
    sample = sample.reshape((120, 320)).astype(np.uint8)
    cv2.imshow('image', sample)
    cv2.waitKey(delay=delay)


if show_images:
    for sample in train:
        show_image(sample)

if export_images:
    for i, sample in enumerate(train):
        sample = sample.reshape((120, 320)).astype(np.uint8)
        cv2.imwrite('export/' + str(i) + '.png', sample)

if not should_train:
    exit('should not train')

e00 = cv2.getTickCount()
time0 = (e00 - e0) / cv2.getTickFrequency()
print 'Loading image duration:', time0

# set start time
e1 = cv2.getTickCount()

if use_tensorflow:
    model = Sequential([
        Reshape((320, 120), input_shape=(train[0].shape)),
        # Conv2D(24, kernel_size=(3, 3), strides=(3, 3), activation='relu', input_shape=input_shape),
        Conv1D(24, kernel_size=3, strides=3, activation='relu'),
        # Conv2D(36, kernel_size=(3, 3), strides=(3, 3), activation='relu'),
        Conv1D(32, kernel_size=3, strides=3, activation='relu'),
        Conv1D(48, kernel_size=3, strides=3, activation='relu'),
        Flatten(),
        Dense(10),
        Dense(4)
    ])

    model.compile(loss='mse', optimizer='adam')

    early_stopping = EarlyStopping(monitor='val_loss')
    history = model.fit(train, train_labels, nb_epoch=120, validation_split=0.2, callbacks=[early_stopping])

    model.save('model.h5')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('history.png')



else:
    # create MLP
    layer_sizes = np.int32([38400, 32, 4])

    model = cv2.ANN_MLP()
    model.create(layer_sizes)
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 0.0001)
    criteria2 = (cv2.TERM_CRITERIA_COUNT, 100, 0.001)
    params = dict(term_crit=criteria,
                  train_method=cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,
                  bp_dw_scale=0.001,
                  bp_moment_scale=0.0)

    print 'Training MLP ...'
    num_iter = model.train(train, train_labels, None, params=params)

    # set end time
    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()
    print 'Training duration:', time

    # save param
    model.save('mlp_xml/mlp.xml')

    print 'Ran for %d iterations' % num_iter

    ret, resp = model.predict(train)
    prediction = resp.argmax(-1)
    print 'Prediction:', prediction
    true_labels = train_labels.argmax(-1)
    print 'True labels:', true_labels

    print 'Testing...'
    train_rate = np.mean(prediction == true_labels)
    print 'Train rate: %f:' % (train_rate * 100)

# try to avoid 'NoneType' object has no attribute 'TF_DeleteStatus' error
keras.backend.clear_session()

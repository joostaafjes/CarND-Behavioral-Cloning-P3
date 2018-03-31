import os
import csv

import cv2
import numpy as np
import sklearn
import sys

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D

import tensorflow as tf
import matplotlib.pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS

root = ''

# command line flags
flags.DEFINE_integer('epochs', 1, "# of epochs")

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("training_results.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        pass

sys.stdout = Logger()

def get_data(cycles):
   lines = []
   set_name = ''
   for cycle in cycles:
       ride = cycle[0]
       track = cycle[1]
       dir = cycle[2]
       set_name += str(ride)

       base_path = root + './data/' + track + '/' + dir + '/' + ride + '/'

       print('Start reading file for ride ' + ride + '...')
       with open(base_path + 'driving_log.csv') as csvfile:
           reader = csv.reader(csvfile)
           for line in reader:
               line.append(base_path)
               lines.append(line)

   return set_name, lines

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            measurements = []
            steering_correction = 0.2
            for line in batch_samples:
                base_path = line[-1]
                image_center = cv2.imread(base_path + './IMG/' + line[0].split('/')[-1])
                image_left = cv2.imread(base_path + './IMG/' + line[1].split('/')[-1])
                image_right = cv2.imread(base_path + './IMG/' + line[2].split('/')[-1])
                images.extend([image_center, image_left, image_right])
                steering_angle = float(line[3])
                measurements.extend([steering_angle, steering_angle + steering_correction, steering_angle - steering_correction])

            # augment with flipped image
            augmented_images = []
            augmented_measurements = []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(np.fliplr(image))
                augmented_measurements.append(measurement * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)

            yield shuffle(X_train, y_train)


"""
If the above code throw exceptions, try 
model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)
"""


def train(set_name, samples):
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    model = Sequential()
    model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # train model
    history_object = model.fit_generator(train_generator, samples_per_epoch=3*len(train_samples), validation_data=validation_generator,
    nb_val_samples=len(validation_samples), nb_epoch=FLAGS.epochs, verbose=2)

    model.save(root + 'model' + set_name + '.h5')

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

def load_and_train(cycles):
    set_name, samples = get_data(cycles)
    train(set_name, samples)

load_and_train([['01', 'track1', 'forw']])
load_and_train([['02', 'track1', 'forw']])
load_and_train([['04', 'track1', 'back']])
load_and_train([['05', 'track1', 'back']])
load_and_train([['01', 'track1', 'forw'],
                ['02', 'track1', 'forw']])
load_and_train([['04', 'track1', 'back'],
                ['05', 'track1', 'back']])
load_and_train([['01', 'track1', 'forw'],
                ['02', 'track1', 'forw'],
                ['04', 'track1', 'back']])
load_and_train([['01', 'track1', 'forw'],
                ['02', 'track1', 'forw'],
                ['05', 'track1', 'back']])
load_and_train([['02', 'track1', 'forw'],
                ['04', 'track1', 'back'],
                ['05', 'track1', 'back']])
load_and_train([['01', 'track1', 'forw'],
                ['02', 'track1', 'forw'],
                ['04', 'track1', 'back'],
                ['05', 'track1', 'back']])
load_and_train([['02', 'track1', 'forw'],
                ['05', 'track1', 'back']])
load_and_train([['02', 'track1', 'forw'],
                ['04', 'track1', 'back']])



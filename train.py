import csv
import cv2
import numpy as np
import sys

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

def get_data_for_ride(ride, track='track1', dir='forw'):
    base_path = root + './data/' + track + '/' + dir + '/' + ride + '/'

    print('Start reading file for ride ' + ride + '...')
    lines = []
    with open(base_path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    steering_correction = 0.2
    for line in lines:
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

    return augmented_images, augmented_measurements

def train(ride, images, measurements):
    X_train = np.array(images)
    y_train = np.array(measurements)

    print('------------------------')
    print('ride:', ride)
    print('------------------------')
    print('min:', np.min(y_train))
    print('max:', np.max(y_train))
    print('mean:', np.mean(y_train))
    print('median:', np.median(y_train))

    model = Sequential()
    model.add(Cropping2D(cropping=((60, 25), (0, 0))))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    # model.add(Flatten())
    # model.add(Dense(1))
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # train model
    history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=FLAGS.epochs, verbose=2)

    model.save(root + 'model' + ride + '.h5')

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

print('using gpu?')
#tf.test.gpu_device_name()
print('No of epochs:' + str(FLAGS.epochs))
images01, measurements01 = get_data_for_ride('01')
images02, measurements02 = get_data_for_ride('02')
images04, measurements04 = get_data_for_ride('04', dir='back')
train('01', images01, measurements01)
train('02', images02, measurements02)
train('04', images04, measurements04)

train('0102', images01 + images02,  measurements01 + measurements02)
train('010204', images01 + images02 + images04, measurements01 + measurements02 + measurements04)


# history
# 1. alleen Flatten en Dense: wielen op neer gaan
# 2. Normalisation: nog steeds
# 3. LeNet alleen: more stable but 'afwijking' naar links
# 4. LeNet and augmenting mirror:


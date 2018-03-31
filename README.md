# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

Below the video of several simulation with different models:

[![Solid white right lane](https://img.youtube.com/vi/iZCi9-zlmgc/0.jpg)](https://www.youtube.com/watch?v=iZCi9-zlmgc)

[//]: # (Image References)

[error_loss]: ./examples/error_loss.png "Mean Squared error loss"

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* train.py containing the script to create and train the model
* train_with_generator.py containing the script to create and train the model by using a generator
* drive.py for driving the car in autonomous mode (unchanged)
* model*.h5 containing a trained convolution neural network for various training runs
* video.py for creating a video from an autonomous run
* join_clips.py to combine several video's in a raster
* (this) README.md summarizing the results

#### 2. Training the model and running the simulator

The following steps have to be done to train the model and run the simulator:

1. Data is gathered with Udacity provided simulator, resulting in -> set of images and csv with ao steering angle
2. CNN models are trainied with data with the train_with_generator.py script, e.g.
```sh
python train_with_generator.py --epochs 5
```
This results in several models by combining all the datasets produced in step 1.

3. Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The train_with_generator.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 1 and 120 (train_with_generator.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 128, 130), and the data is normalized in the model using a Keras lambda layer (code line 127). Also the top and bottom of the input images is trimmed (line 126) 

#### 2. Attempts to reduce overfitting in the model

The model contains no dropout layers because it looked like overfitting was not occuring: accuracy on training set was close to validation accuracy. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (training_with_generator.py line 137).

#### 4. Mean squared error loss

The figure below shows the mean squared error loss during the training of the model.

![error_loss][error_loss]

#### 5. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Also I drove the track in reverse order.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet architecture used in previous lessons.

Initial the model was not good enough, so took the following steps to improve the model:
1. Normalisation of the input image to [-0.5, 0,5]
2. Trim the bottom and lower part of the image, so only the real track was visible
3. Augmenting the input by adding the left and right images and add them with a corrected steering angle

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Creation of the Training Set & Training Process

After these steps the model was able to drive stable at the track except at some spots, but not through the whole track. To overcome this, I created 4 tracksets: 2 forward and 2 backwords. The quality of these sets was not very stable cause I am a lousy driver. 
I trained the model with different combinations of the dataset to get the best combination. 

Finally the dataset 2, 4 and 5 together gave the best results (see attached YouTube video). Appearantly dataset 1 was of bad quality.




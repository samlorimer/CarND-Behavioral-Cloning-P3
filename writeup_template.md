# **Behavioral Cloning Project** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia-cnn-architecture.png "Model Visualization"
[image2]: ./examples/center_example.jpg "Center Camera example"
[image3]: ./examples/left_example.jpg "Left Camera example"
[image4]: ./examples/right_example.jpg "Right Camera example"
[image5]: ./examples/recovery_1.jpg "Recovery Image 1"
[image6]: ./examples/recovery_2.jpg "Recovery Image 2"
[image7]: ./examples/recovery_3.jpg "Recovery Image 3"
[image8]: ./examples/recovery_3_normal.jpg "Flipped image"
[image9]: ./examples/recovery_3_flipped.jpg "Flipped image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md (identical to this readme!) summarizing the results
* video.mp4 showing the car driving one autonomous lap successfully on track 1

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a replica of the NVidia convolution neural network for self driving cars (https://devblogs.nvidia.com/deep-learning-self-driving-cars/) and as pictured below:

![alt text][image1]

The model includes RELU layers to introduce nonlinearity after each convolutional layer, and the data is normalized in the model using a Keras lambda layer (code line 62). 

#### 2. Attempts to reduce overfitting in the model

The model uses dropout  in order to reduce overfitting (model.py lines 64 and beyond). A lower probability of dropping was used on the early convolutional layers to avoid losing too much data, and more aggressive dropout was applied on the fully-connected layers.

The model was trained and validated on different data sets (shuffled and split before each training session) to ensure that the model was not overfitting (code line 86). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 85).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, driving arond the track in reverse along with recovering from the left and right sides of the road.

For details about how I created the training data, see section 7. 

#### 5. Solution Design Approach

The overall strategy for deriving a model architecture was to build up slowly to a complex model and see how the driving behaviour improved on my dataset.

My first step was to use a convolution neural network model similar to LeNet.  I thought this model might be appropriate because it performs well on image classification, and is very quick and easy to implement from previous work in this course.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set while training using keras. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.  I also found that the model didn't drive around the track terribly effectively!

To improve the driving, I then changed the architecture to match the NVidia self-driving architecture.  To tackle the overfitting, I modified the model so that dropout was applied more aggressively, also detailed above.  I also ensured the image files were being used in the same colour format for training as they were by drive.py when running the model.

This improved the MSE difference between the training and validation set and also saw the car begin to successfully drive around most of the track.

There were a few spots where the vehicle fell off the track - mostly on the bridge and the two turns following (one with a dirt border next to the track, and the next a tight right-hand turn).  To improve the driving behavior in these cases, I augmented the data set by flipping all images and reversing the steering measurement.  I also trained additional driving data as detailed below.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 6. Final Model Architecture

The final model architecture (model.py lines 61-81) consisted of a convolution neural network matching the NVidia architecture (detailed above).  The decision to use dropout to reduce overfitting and rates of dropout applied are also detailed above.

#### 7. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then used the left and right cameras to augment this data so that not all driving appeared to be in the center of the track.  An offset of 0.2 (and -0.2) was added to the steering measurement for the left and right images to encourage the model to steer back to the centre.  Here is an example of left & right camera images:

![alt text][image3]
![alt text][image4]

I then recorded the vehicle recovering from the left side and right sides of the road back to center to further encourage the model to allow the vehicle to steer more sharply and resume central driving.  Here is an example of what a recovery recording consisted of, steering from the left of the track back to the centre:

![alt text][image5]
![alt text][image6]
![alt text][image7]

Then I repeated this process whilst driving the track in reverse to gather even more data points and attempt to generalise the driving style.

To augment the data sat, I also flipped images captured so that the model would learn equally to steer and manouver around corners to the left and right.  For example, here is an original image, followed by the same that has then been flipped and the steering angle reversed in the data set:

![alt text][image8]
![alt text][image9]

After the collection process, I had 25065 data points. I then preprocessed this data as described above (normalising pixel values and then cropping the image by 70 pixels at the top and 25 at the bottom to remove the sky and hood of the car).

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the validation loss plateauing at that point and the vehicle becoming capable of completing a full lap autonomously.  I used an adam optimizer so that manually training the learning rate wasn't necessary.

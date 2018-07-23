import csv
import cv2
import numpy as np

# Read the driving log file to list
lines = []
with open('../driving_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

# correction factor indicates the adjustment to the steering data to apply to 
# the left & right camera images
correction_factor = 0.2

# Now read each image and its corresponding measurement into their respective lists
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = '../driving_data/IMG/' + filename
        image = cv2.imread(current_path)
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(imageRGB)
        measurement = float(line[3])
        # Use the left [1] & right [2] camera images with correction factor applied
        if i == 1:
            measurement+=correction_factor
        elif i == 2:
            measurement-=correction_factor
        measurements.append(measurement)

# Now add a flipped version of all images with the inverse measurement to
# augment the data set
augmented_images, augmented_measurements = [], []
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

# Convert lists to numpy arrays for use with Keras
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# import the necessary Keras frameworks
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Implement the NVidia self-driving architecture as defined here:
# https://devblogs.nvidia.com/deep-learning-self-driving-cars/
# In addition, we are using a lambda function to normalise the pixel values
# in each image, and crop the sky and hood of the car out of each image.
# Finally, dropout layers have been introduced - dropping with a rate of
# 0.25 for convolutional layers, and 0.5 for fully-connected layers
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.25))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.25))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.25))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.25))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(50,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1))

# Use mean-squared-error ,loss function and adam optimiser, and
# shuffle the data before running with a 80/20 train/validation split
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')


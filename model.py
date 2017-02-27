import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model

# Read the log file and append each line into an array
initial_lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        initial_lines.append(line)
        
# Remove the first element in the array for line        
initial_lines.pop(0)

# Create a better distribution for the measurements
new_lines = []
zero_lines = []

for line in initial_lines:
    if line[3] == " 0":
        zero_lines.append(line)
    else:
        new_lines.append(line)

lines = new_lines + zero_lines[:2600]

# Get all the images and append each image into an array
# Get all measurments and append each measurement into an array
images = []
measurements = []
for line in lines:
    source_path = line[0]
    tokens = source_path.split('/')
    filename = tokens[-1]
    local_path = "./data/IMG/" + filename
    image = cv2.imread(local_path)
    images.append(image)
    measurement = line[3]
    measurements.append(measurement)

# Set up training data into features and labels
X_train = np.array(images)
y_train = np.array(measurements)

#################### CNN START ############################
# Create Convolutional Neural Network
model = Sequential()

# Normalise the data
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# Layer 1 - Convolutional
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation = 'relu'))

# Layer 2 - Convolutional 
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation = 'relu'))

# Layer 3 - Convolutional
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation = 'relu'))

# Layer 4 - Convolutional
model.add(Convolution2D(64, 3, 3, activation = 'relu'))

# Layer 5 - Convolutional
model.add(Convolution2D(64, 3, 3, activation = 'relu'))

# Layer 6 - Fully Connected
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.25))

# Layer 7 - Fully Connected
model.add(Dense(50))
model.add(Dropout(0.25))

# Layer 8 - Fully Connected
model.add(Dense(10))

# Layer 9 - Fully Connected
model.add(Dense(1))
#################### CNN END ############################

# Model compilation - Configure the learning process
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, validation_split=0.3, shuffle=True, nb_epoch = 5)

# Save model
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
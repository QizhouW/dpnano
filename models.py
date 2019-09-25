"""
A collection of models we'll use to attempt to classify videos.
"""
import sys
from keras import layers
from keras import models
# Import necessary packages
import argparse
# Import necessary components to build LeNet
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import load_model, Model

class ResearchModels():
    def __init__(self, model, npoints=20, saved_model=None):
        # Now compile the network.
        self.nb_of_points = npoints
        self.saved_model = saved_model
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model=='alexnet':
            print("Loading alexnet model.")
            self.model = self.alexnet()
        elif model=='primanet':
            print("Loading prima model.")
            self.model = self.primanet()
        elif model=='primanet2':
            print("Loading prima 2 model.")
            self.model = self.primanet2()
        else:
            print("Unknown network.")
            sys.exit()

        optimizer = Adam(lr=1e-5, decay=1e-6)
        self.model.compile(loss='mse', optimizer=optimizer,metrics=['mse'])
        print(self.model.summary())

    def alexnet(self):
        # (3) Create a sequential model
        model = Sequential()

        # 1st Convolutional Layer
        model.add(Conv2D(filters=96, input_shape=(250, 250, 3), kernel_size=(11, 11), \
                         strides=(4, 4), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        # Batch Normalisation before passing it to the next layer
        model.add(BatchNormalization())

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 4th Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 5th Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(4, 4), strides=(2, 2), padding='valid'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # Passing it to a dense layer
        model.add(Flatten())
        # 1st Dense Layer
        model.add(Dense(512))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 2nd Dense Layer
        model.add(Dense(512))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))
        # Batch Normalisation

        # Output Layer
        model.add(Dense(self.nb_of_points))
        model.add(Activation('sigmoid'))
        return model

    def primanet(self):
        model = Sequential()
        model.add(Conv2D(32, (11, 11), activation='relu',
                         input_shape=(250, 250, 1)))
        model.add(MaxPooling2D((3, 3)))
        model.add(Conv2D(64, (11, 11), activation='relu'))
        model.add(MaxPooling2D((3, 3)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (2, 2), activation='relu'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_of_points, activation='sigmoid'))
        return model

    def primanet2(self):
        model = Sequential()
        model.add(Conv2D(64, (11, 11), activation='relu',
                         input_shape=(250, 250, 3)))
        model.add(MaxPooling2D((3, 3)))
        model.add(Conv2D(128, (11, 11), activation='relu'))
        model.add(MaxPooling2D((3, 3)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (2, 2), activation='relu'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_of_points, activation='sigmoid'))
        return model

    def primanet3(self):
        pass






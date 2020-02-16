# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 21:31:04 2020

@author: Tanmay Thakur
"""

import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, SeparableConv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model


def vanilla_conv(image_x,image_y):
    num_of_classes = 37
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

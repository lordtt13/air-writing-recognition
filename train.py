# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 22:30:12 2020

@author: Tanmay Thakur
"""
import pandas as pd
import numpy as np

from tensorflow.keras.utils import to_categorical
from model import *


data = pd.read_csv("data.csv")
dataset = np.array(data)
np.random.shuffle(dataset)
X = dataset
Y = dataset
X = X[:, 0:1024]
Y = Y[:, 1024]

image_x = 32
image_y = 32

Y = to_categorical(Y)
X = np.reshape(X, (X.shape[0], image_x, image_y, 1))

model_new = vanilla_conv(image_x, image_y)

model_new.fit(X, Y, batch_size = 32, epochs = 100, validation_split =  0.25)

# model_sc = sc_conv(image_x, image_y)

# model_complex = sc_conv_complex(image_x, image_y)
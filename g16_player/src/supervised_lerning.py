#!/usr/bin/env python3
import numpy as np

import tensorflow as tf
import numpy as np
from keras import layers as l
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Conv2DTranspose, BatchNormalization, Dropout, Flatten, concatenate
from keras.models import Model
from keras.utils import plot_model, get_file
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# supervised learning model 

a = np.load("inputs.npy")
b = np.load("outputs.npy")

inputs = Input(shape=(25,))
x = Dense(20, activation='relu')(inputs)
outputs = Dense(2, activation='linear')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.fit(a, b, )

print(a.shape, b.shape)
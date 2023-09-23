# -*- coding: utf-8 -*-
"""IU_NET_model.ipynb

# ** Inception U-Net Model = (Inception Blocks embeded only at Encoder path) **

## **Importing Libraries**
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, MaxPooling3D, BatchNormalization, Activation, Concatenate
from tensorflow.keras.models import Model

import numpy as np
import nibabel as nib
import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tifffile import imsave
import random
import tensorflow.keras.backend as K

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from tensorflow.keras import models, layers, regularizers
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from keras.metrics import MeanIoU
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout,Maximum

"""# **IU-Net Architecture With 5 Levels**"""

def conv_block(input_mat):
  num_filters = 32
  kernel_size = 3
  batch_norm = True

  X = Conv3D(num_filters,kernel_size=(kernel_size,kernel_size,kernel_size),strides=(1,1,1),padding='same')(input_mat)
  X = BatchNormalization()(X)
  X = Activation('leaky_relu')(X)

  X = Conv3D(num_filters,kernel_size=(kernel_size,kernel_size,kernel_size),strides=(1,1,1),padding='same')(X)
  X = BatchNormalization()(X)
  X = Activation('leaky_relu')(X)

  return X

def inception_block(inputs):
    n_filters = 32
    conv1x1 = Conv3D(n_filters, kernel_size=1, activation='relu', padding='same')(inputs)

    conv3x3 = Conv3D(n_filters, kernel_size=1, activation='relu', padding='same')(inputs)
    conv3x3 = Conv3D(n_filters, kernel_size=3, activation='relu', padding='same')(conv3x3)

    conv5x5 = Conv3D(n_filters, kernel_size=1, activation='relu', padding='same')(inputs)
    conv5x5 = Conv3D(n_filters, kernel_size=5, activation='relu', padding='same')(conv5x5)

    maxpool3x3 = MaxPooling3D(pool_size=(3, 3, 3), strides=1, padding='same')(inputs)
    maxpool3x3 = Conv3D(n_filters, kernel_size=1, activation='relu', padding='same')(maxpool3x3)

    output = concatenate([conv1x1, conv3x3, conv5x5, maxpool3x3], axis=-1)
    return output

# Define 3D Attention Inception U-Net model with attention blocks only at the encoder part
def inception_unet(input_shape):
    # Input Layer
    inputs = Input(shape=input_shape)

    # Level 1 (Encoder)
    enc_inception1 = inception_block(inputs)
    enc_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(enc_inception1)

    # Level 2

    enc_inception2 = inception_block(enc_pool1)
    enc_pool2 = MaxPooling3D(pool_size=(2, 2, 2))(enc_inception2)

    # Level 3

    enc_inception3 = inception_block(enc_pool2)
    enc_pool3 = MaxPooling3D(pool_size=(2, 2, 2))(enc_inception3)

    # Level 4

    enc_inception4 = inception_block(enc_pool3)
    enc_pool4 = MaxPooling3D(pool_size=(2, 2, 2))(enc_inception4)

    # Level 5 (Bridge)
    bridge_conv = Conv3D(filters=1024, kernel_size=(3, 3, 3), padding='same', activation='relu')(enc_pool4)
    bridge_inception = inception_block(bridge_conv)

    # Level 4 (Decoder)

    dec_upconv4 = Conv3DTranspose(filters=512, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(bridge_inception)
    dec_concat4 = Concatenate(axis=-1)([dec_upconv4, enc_inception4])
    dec_inception4 = conv_block(dec_concat4)
    #print("level 4 DEc",dec_inception4.shape)

    # Level 3
    dec_upconv3 = Conv3DTranspose(filters=256, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(dec_inception4)
    dec_concat3 = Concatenate(axis=-1)([dec_upconv3, enc_inception3])
    dec_inception3 = conv_block(dec_concat3)

    # Level 2
    dec_upconv2 = Conv3DTranspose(filters=128, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(dec_inception3)
    dec_concat2 = Concatenate(axis=-1)([dec_upconv2, enc_inception2])
    dec_inception2 = conv_block(dec_concat2)

    # Level 1
    dec_upconv1 = Conv3DTranspose(filters=64, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(dec_inception2)
    dec_concat1 = Concatenate(axis=-1)([dec_upconv1, enc_inception1])
    dec_inception1 = conv_block(dec_concat1)

    # Output
    output = Conv3D(filters=4, kernel_size=(1, 1, 1), activation='softmax')(dec_inception1)

    model = Model(inputs=inputs, outputs=output)
    return model

input_shape = (128, 128, 128, 4) # Adjust the input shape according to data
model = inception_unet(input_shape)

print(model.input)

print(model.output)

model.summary()

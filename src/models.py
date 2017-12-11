import os
import sys
import glob
import shutil
import random
from random import randint
import numpy as np 
from PIL import Image
from PIL import ImageFilter


import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Reshape, Input
from keras.layers.core import Activation, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3


# Cyclical learning rates 
def triangular2(epoch):
    base_lr = .0005
    max_lr = .005
    step_size = 5

    cycle = np.floor(1+epoch/(2*step_size))
    x = np.abs(epoch/step_size - 2*cycle + 1)
    lr = base_lr + (max_lr-base_lr)*np.maximum(0, (1-x))/float(2**(cycle-1))
    return lr

# A flatten that doesn't cause problems when input shape isn't defined:
def flatten(x):
    batch_size = K.shape(x)[0]
    x = K.reshape(x, (batch_size, -1)) 
    return x

def conv_bn_dp(x, filters, dropout):
    a = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    b = BatchNormalization()(a)
    c = Activation('relu')(b)
    d = Dropout(dropout)(c)
    return d

# Basic medium sided CNN with fully connected layer
def conv_6L(learning_rate = .001, dropout = .1, num_output=2):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same', input_shape=(512, 512, 3), kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    
    model.add(Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    
    model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    
    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(512, kernel_initializer="he_normal"))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout*5))

    model.add(Dense(num_output, kernel_initializer="he_normal"))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('softmax'))

    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model


####.    https://github.com/jacobgil/keras-cam/blob/master/model.py
from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras import backend as K
import h5py

def global_average_pooling(x):
    return K.mean(x, axis = (2, 3))

def global_average_pooling_shape(input_shape):
    return input_shape[0:2]

def VGG16_convolutions():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(None,None, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_3'))
    return model

def get_vgg_var_sz(weights_path, learning_rate=.001, dropout = .1, num_output=2, img_dim=512):
    model = VGG16_convolutions()

    model = load_model_weights(model, weights_path)
    
    model.add(Lambda(global_average_pooling, 
              output_shape=global_average_pooling_shape))
    model.add(Dense(num_output, activation = 'softmax', init='uniform'))
    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam, metrics=['accuracy'])
    return model

def load_model_weights(model, weights_path):
    print('Loading model.')
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
        model.layers[k].trainable = False
    f.close()
    print('Model loaded.')
    return model

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


#################################





# Pre-trained VGG model using 2 fully connected layers. 
# Standard for baseline performance
def vgg16_1(learning_rate=.001, dropout = .1, num_output=2, img_dim=224):
    # try:
    #     vgg16 = VGG16(weights='imagenet', include_top=False,  input_shape = (512, 512, 3))
    # except:
    vgg16  = load_model('project/rbbidart/models/vgg16')
    x=vgg16.get_layer('block5_pool').output
    
    x = Flatten(input_shape=(int(img_dim/32), int(img_dim/32), 512))(x) # last conv layer outputs 7x7x512
    x = Dense(4096, kernel_initializer='glorot_uniform')(x)
    x = keras.layers.normalization.BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(512, kernel_initializer='glorot_uniform')(x)
    x = keras.layers.normalization.BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(num_output, kernel_initializer='glorot_uniform')(x)
    x = keras.layers.normalization.BatchNormalization()(x)
    pred = Activation('softmax')(x)
    
    model = Model(outputs = pred, inputs = vgg16.input)
    
    for layer in model.layers[:19]: # only train fc layers. 
        layer.trainable = False

    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model

# Fully convolutional VGG-16. 3 conv layers instead of fully connected, followed by global pooling
# ???? FC1 doesn't downsample, so cna be used with normal 224 sized VVG
def vgg16_fc1(learning_rate=.001, dropout = .1, num_output=2, img_dim=224): # have to leave size = 224
    # try:
    #     vgg16 = VGG16(weights='imagenet', include_top=False)
    # except:
    vgg16  = load_model('project/rbbidart/models/vgg16')
    x=vgg16.get_layer('block5_pool').output

    x = Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=(int(img_dim/32), int(img_dim/32), 512))(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)

    x = Conv2D(num_output, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    pred = Activation('softmax')(x)
    
    model = Model(outputs = pred, inputs = vgg16.input)
    
    for layer in model.layers[:19]: # only train fc layers. 
        layer.trainable = False

    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model

# Fully convolutional VGG-16. 2 conv layers, global pooling, dense to rehape to proper class number
def vgg16_fc1b(learning_rate=.001, dropout = .1, num_output=2, img_dim=224):
    # try:
    #     vgg16 = VGG16(weights='imagenet', include_top=False)
    # except:
    vgg16  = load_model('project/rbbidart/models/vgg16')
    x=vgg16.get_layer('block5_pool').output

    x = Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=(int(img_dim/32), int(img_dim/32), 512))(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(num_output, activation = 'softmax')(x)
    pred = Activation('softmax')(x)
    
    model = Model(outputs = pred, inputs = vgg16.input)
    
    for layer in model.layers[:19]: # only train fc layers. 
        layer.trainable = False

    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model

# try adding one more conv layer before max pool
def vgg16_fc2(learning_rate=.001, dropout = .1, num_output=2, img_dim=224):
    # same freatures as include_top=False
    # try:
    #     vgg16 = VGG16(weights='imagenet', include_top=False)
    # except:
    vgg16  = load_model('project/rbbidart/models/vgg16')
    x=vgg16.get_layer('block5_conv3').output

    x = Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=(int(img_dim/16), int(img_dim/16), 512))(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(num_output, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    pred = Activation('softmax')(x)
    
    model = Model(outputs = pred, inputs = vgg16.input)
    
    for layer in model.layers[:19]: # only train fc layers. 
        layer.trainable = False

    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model



##### Inception





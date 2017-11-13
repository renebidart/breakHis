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

def vgg16_ft(learning_rate=.001, dropout = .1, num_output=2):
    # same freatures as include_top=False
    try:
        vgg16 = VGG16(weights='imagenet', include_top=False,  input_shape = (512, 512, 3))
    except:
        vgg16  = load_model('project/rbbidart/models/vgg16')
    x=vgg16.get_layer('block5_pool').output
    
    x = Flatten(input_shape=(16, 16, 512))(x) # last conv layer outputs 7x7x512
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


# FC1 doesn't downsample, so cna be used with normal 224 sized VVG
def vgg16_fc1(learning_rate=.001, dropout = .1, num_output=2):
    # same freatures as include_top=False
    try:
        vgg16 = VGG16(weights='imagenet', include_top=False)
    except:
        vgg16  = load_model('project/rbbidart/models/vgg16')
    x=vgg16.get_layer('block5_pool').output

    x = Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=(16, 16, 512))(x)
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

def vgg16_fc1b(learning_rate=.001, dropout = .1, num_output=2):
    # same freatures as include_top=False
    try:
        vgg16 = VGG16(weights='imagenet', include_top=False)
    except:
        vgg16  = load_model('project/rbbidart/models/vgg16')
    x=vgg16.get_layer('block5_pool').output

    x = Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=(16, 16, 512))(x)
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


def vgg16_fc2(learning_rate=.001, dropout = .1, num_output=2):
    # same freatures as include_top=False
    try:
        vgg16 = VGG16(weights='imagenet', include_top=False)
    except:
        vgg16  = load_model('project/rbbidart/models/vgg16')
    x=vgg16.get_layer('block5_pool').output

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
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





############### wierd models:

def conv_6L_CSD(im_size, learning_rate = .01, dropout = .1):

    def CSD_loss(y_true, y_pred):
        # always assume there will be batch size of 4, with the elements in the following order:
        # [current element, element in same sub class, element in same class, element in different class]
        m1 = .05 # no idea what this should be set to
        m2 = .1

        # Normal loss function for CNN:
        loss_classification = K.categorical_crossentropy(y_true, y_pred)

        # Loss function based on feature distances:
        # after the flatten dims is [batches, features]
        d_x_pos = K.sqrt(K.sum(K.square(dense2[0, :] - dense2[1, :])))

        d_x_neg = K.sqrt(K.sum(K.square(dense2[1, :] - dense2[2, :])))
        d_x_n = K.sqrt(K.sum(K.square(dense2[2, :] - dense2[3, :])))

        # loss_feat_dist = (.5*K.mean(K.max(0, d_x_pos - d_x_neg + m1 - m2)) + .5*K.mean(K.max(0, d_x_neg - d_x_n + m2)))
        # Can't use the above because it ids for the case with multiple samples of each class
        loss_feat_dist = .5*K.maximum(0.0, d_x_pos - d_x_neg + m1 - m2) + .5*K.maximum(0.0, d_x_neg - d_x_n + m2)    
        loss = 0.5*loss_classification + 0.5*loss_feat_dist
        
        return loss
    
    input_shape = (im_size, im_size, 3)
    img_input = Input(shape=input_shape)

    L1 = conv_bn_dp(img_input, filters=16, dropout=dropout)

    L2 = conv_bn_dp(L1, filters=16, dropout=dropout)
    L2_pool = MaxPooling2D(pool_size=(2, 2))(L2)

    L3 = conv_bn_dp(L2, filters=32, dropout=dropout)
    
    L4 = conv_bn_dp(L3, filters=32, dropout=dropout)
    L4_pool = MaxPooling2D(pool_size=(2, 2))(L4)

    L5 = conv_bn_dp(L4_pool, filters=64, dropout=dropout)
    
    L6 = conv_bn_dp(L5, filters=64, dropout=dropout)
    L5_pool = MaxPooling2D(pool_size=(2, 2))(L5)
    
    flatten = Flatten(input_shape=(None, None, 64) )(L5_pool)
    
    dense1 = Dense(512, kernel_initializer='he_normal')(flatten)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense1 = Dropout(dropout*4)(dense1)

    dense2 = Dense(256, kernel_initializer='he_normal')(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    dense2 = Dropout(dropout*4)(dense2)
    
    predictions = Dense(8, activation='softmax')(dense2)
    model = Model(outputs=predictions, inputs=img_input)

    SGD = keras.optimizers.SGD(lr=learning_rate, momentum=0.3, decay=0.0, nesterov=False)
    model.compile(optimizer=SGD, loss=CSD_loss, metrics=['accuracy'])
    return model


def InceptionV3_CSD(im_size, learning_rate=.001, dropout =.1):

    def CSD_loss(y_true, y_pred):
        # always assume there will be batch size of 4, with the elements in the following order:
        # [current element, element in same sub class, element in same class, element in different class]
        m1 = .05 # no idea what this should be set to
        m2 = .1

        # Normal loss function for CNN:
        loss_classification = K.categorical_crossentropy(y_true, y_pred)

        # Loss function based on feature distances:
        # after the flatten dims is [batches, features]
        d_x_pos = K.sqrt(K.sum(K.square(dense1_out[0, :] - dense1_out[1, :])))

        d_x_neg = K.sqrt(K.sum(K.square(dense1_out[1, :] - dense1_out[2, :])))
        d_x_n = K.sqrt(K.sum(K.square(dense1_out[2, :] - dense1_out[3, :])))

        # loss_feat_dist = (.5*K.mean(K.max(0, d_x_pos - d_x_neg + m1 - m2)) + .5*K.mean(K.max(0, d_x_neg - d_x_n + m2)))
        # Can't use the above because it ids for the case with multiple samples of each class
        loss_feat_dist = .5*K.maximum(0.0, d_x_pos - d_x_neg + m1 - m2) + .5*K.maximum(0.0, d_x_neg - d_x_n + m2)    
        loss = 0.5*loss_classification + 0.5*loss_feat_dist
        return loss

    # try:
    base_model = InceptionV3(weights='imagenet', include_top=False)
    # except:
    #     InceptionV3  = load_model('project/rbbidart/models/InceptionV3')
    #     base_model = InceptionV3(inp)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    dense1 = Dense(1024, kernel_initializer='he_normal')(x)
    dense1 = keras.layers.normalization.BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense1_out = Dropout(dropout*4)(dense1)

    # Output layer
    predictions = Dense(8, activation='softmax')(dense1_out)

    # add everything together to get a model.
    model = Model(outputs=predictions, inputs=base_model.input)

    # This freezes the convolutional layers, so only the added FC layers will be trained
    # Can still adjust the base model because it hasn't been compiled yet
    for layer in base_model.layers:
        layer.trainable = False

    SGD = keras.optimizers.SGD(lr=learning_rate, momentum=0.3, decay=0.0, nesterov=False)
    model.compile(optimizer=SGD, loss=CSD_loss, metrics=['accuracy'])
    return model


def InceptionV3_csd_noL(im_size, learning_rate=.01, dropout =.1, num_output=8):
    # try:
    base_model = InceptionV3(weights='imagenet', include_top=False)
    # except:
    #     InceptionV3  = load_model('project/rbbidart/models/InceptionV3')
    #     base_model = InceptionV3(inp)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    dense1 = Dense(1024, kernel_initializer='he_normal')(x)
    dense1 = keras.layers.normalization.BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense1_out = Dropout(dropout*4)(dense1)

    # Output layer
    predictions = Dense(8, activation='softmax')(dense1_out)

    # add everything together to get a model.
    model = Model(outputs=predictions, inputs=base_model.input)

    # This freezes the convolutional layers, so only the added FC layers will be trained
    # Can still adjust the base model because it hasn't been compiled yet
    for layer in base_model.layers:
        layer.trainable = False

    SGD = keras.optimizers.SGD(lr=learning_rate, momentum=0.3, decay=0.0, nesterov=False)
    model.compile(optimizer=SGD, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def InceptionV3_CSD2(im_size, learning_rate=.001, dropout =.1):

    def CSD_loss(y_true, y_pred):
        # always assume there will be batch size of 4, with the elements in the following order:
        # [current element, element in same sub class, element in same class, element in different class]
        m1 = .05 # no idea what this should be set to
        m2 = .1

        # Normal loss function for CNN:
        loss_classification = K.categorical_crossentropy(y_true, y_pred)

        # Loss function based on feature distances:
        # after the flatten dims is [batches, features]
        d_x_pos = K.sqrt(K.sum(K.square(dense1_out[0, :] - dense1_out[1, :])))

        d_x_neg = K.sqrt(K.sum(K.square(dense1_out[1, :] - dense1_out[2, :])))
        d_x_n = K.sqrt(K.sum(K.square(dense1_out[2, :] - dense1_out[3, :])))

        # loss_feat_dist = (.5*K.mean(K.max(0, d_x_pos - d_x_neg + m1 - m2)) + .5*K.mean(K.max(0, d_x_neg - d_x_n + m2)))
        # Can't use the above because it ids for the case with multiple samples of each class
        loss_feat_dist = .5*K.maximum(0.0, d_x_pos - d_x_neg + m1 - m2) + .5*K.maximum(0.0, d_x_neg - d_x_n + m2)    
        loss = 0.5*loss_classification + 0.5*loss_feat_dist
        return loss

    # try:
    base_model = InceptionV3(weights='imagenet', include_top=False)
    # except:
    #     InceptionV3  = load_model('project/rbbidart/models/InceptionV3')
    #     base_model = InceptionV3(inp)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    dense1 = Dense(1024, kernel_initializer='he_normal')(x)
    dense1 = keras.layers.normalization.BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense1_out = Dropout(dropout*4)(dense1)

    # Output layer
    predictions = Dense(8, activation='softmax')(dense1)

    # add everything together to get a model.
    model = Model(outputs=predictions, inputs=base_model.input)

    # This freezes the convolutional layers, so only the added FC layers will be trained
    # Can still adjust the base model because it hasn't been compiled yet
    for layer in base_model.layers:
        layer.trainable = False

    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model

def InceptionV3_CSD3(im_size, learning_rate=.001, dropout =.1):

    def CSD_loss(y_true, y_pred):
        # always assume there will be batch size of 4, with the elements in the following order:
        # [current element, element in same sub class, element in same class, element in different class]
        m1 = .05 # no idea what this should be set to
        m2 = .1

        # Normal loss function for CNN:
        loss_classification = K.categorical_crossentropy(y_true, y_pred)

        # Loss function based on feature distances:
        # after the flatten dims is [batches, features]
        d_x_pos = K.sqrt(K.sum(K.square(dense1_out[0, :] - dense1_out[1, :])))

        d_x_neg = K.sqrt(K.sum(K.square(dense1_out[1, :] - dense1_out[2, :])))
        d_x_n = K.sqrt(K.sum(K.square(dense1_out[2, :] - dense1_out[3, :])))

        # loss_feat_dist = (.5*K.mean(K.max(0, d_x_pos - d_x_neg + m1 - m2)) + .5*K.mean(K.max(0, d_x_neg - d_x_n + m2)))
        # Can't use the above because it ids for the case with multiple samples of each class
        loss_feat_dist = .5*K.maximum(0.0, d_x_pos - d_x_neg + m1 - m2) + .5*K.maximum(0.0, d_x_neg - d_x_n + m2)    
        loss = 0.5*loss_classification + 0.5*loss_feat_dist
        return loss

    # try:
    inp = Input(shape=(int(im_size), int(im_size), 3))

    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model(inp)
    # except:
    #     InceptionV3  = load_model('project/rbbidart/models/InceptionV3')
    #     base_model = InceptionV3(inp)

    dense1 = Flatten(input_shape= (8, 8, 2048) )(x)
    dense1 = Dense(1024, kernel_initializer='he_normal')(dense1)
    dense1 = keras.layers.normalization.BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense1_out = Dropout(dropout*4)(dense1)

    # Output layer
    predictions = Dense(8, activation='softmax')(dense1_out)

    # add everything together to get a model.
    model = Model(outputs=predictions, inputs=inp)

    # This freezes the convolutional layers, so only the added FC layers will be trained
    # Can still adjust the base model because it hasn't been compiled yet
    for layer in base_model.layers:
        layer.trainable = False

    Adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model


#### DATA GENERATORS
def data_gen(file_loc, batch_size, return_binary=False, magnification=0, im_size=None, 
    square_rot_p=.3, translate=0, flips=False, rotate=False, blur=False):
    # square_rot_p is the prob of using a 90x rotation, otherwise sample from 360. Possibly not useful
    # translate is maximum number of pixels to translate by. Make it close the doctor's variance in annotation

    label_list=['B_A', 'B_F', 'B_PT', 'B_TA', 'M_DC', 'M_LC', 'M_MC', 'M_PC']

    square_rot_p = float(square_rot_p)
    translate = int(translate)

    all_files=glob.glob(os.path.join(file_loc, '*'))
    if int(magnification)!=0:
        all_files = [loc for loc in all_files if loc.rsplit('/', 1)[1].rsplit('-', 1)[0].rsplit('-', 1)[1] == str(magnification)]

    while 1:
        random.shuffle(all_files) # randomize after every epoch
        num_batches = int(np.floor(len(all_files)/batch_size))

        for batch in range(num_batches):
            x=[]
            y=[]
            batch_files = all_files[batch_size*batch:batch_size*(batch+1)]
            for image_loc in batch_files:
                image = Image.open(image_loc)

                # APPLY AUGMENTATION:
                # flips
                if flips:
                    flip_vert = random.randint(0, 1)
                    flip_hor = random.randint(0, 1)
                    if flip_vert:
                        image = image.transpose(Image.FLIP_TOP_BOTTOM)
                    if flip_hor:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)

                # rotation
                if rotate:
                    square_rot =  bool((np.random.uniform(0, 1, 1)<square_rot_p))
                    if square_rot:  # maybe this is dumb, but it cant hurt
                        angle = random.randint(0, 4)
                        if(angle ==0):
                            image = image.transpose(Image.ROTATE_90)
                        elif(angle ==1):
                            image = image.transpose(Image.ROTATE_180)
                        elif(angle ==2):
                            image = image.transpose(Image.ROTATE_270)
                    else:
                        angle = np.random.uniform(0, 360,1)
                        image=image.rotate(angle)

                #blur
                if blur:
                    if random.randint(0, 1):
                        radius = random.randint(1, 4)
                        image = image.filter(ImageFilter.GaussianBlur(radius=radius))

                if(im_size != 0):
                    image_shape = (im_size, im_size)
                    image = image.resize(image_shape)

                # translate
                ts_sz_row = randint(-1*translate, translate)
                ts_sz_col = randint(-1*translate, translate)

                image = image.transform(image.size, Image.AFFINE, (1, 0, ts_sz_row, 0, 1, ts_sz_col))

                image = np.reshape(np.array(image.getdata()), (im_size, im_size, 3))
                image = image/255.0 


                label = image_loc.rsplit('/', 1)[1].split('_', 1)[1].split('-', 1)[0]
                if return_binary:
                    y_temp = int(label in label_list[4:])
                    y_temp = np.eye(2)[y_temp]
                else :
                    y_temp = label_list.index(label)
                    y_temp = np.eye(8)[y_temp]
                
                x.append(image)
                y.append(y_temp)
            x=np.array(x)
            y=np.array(y)
            yield (x, y)


# this must return 4 images, from the correct classes in the correct order:
# [current element, element in same sub class, element in same class, element in different class]
# this should also sample even numbers from each class (same number of subclasses for malignant and benign)
def data_gen_CSD(file_loc, batch_size, magnification=0, im_size=None,
    square_rot_p=.3, translate=0, flips=False, rotate=False):
    # square_rot_p is the prob of using a 90x rotation, otherwise sample from 360. Possibly not useful
    # translate is maximum number of pixels to translate by. Make it close the doctor's variance in annotation
    label_list = ['B_A', 'B_F', 'B_PT', 'B_TA', 'M_DC', 'M_LC', 'M_MC', 'M_PC']
    square_rot_p = float(square_rot_p)
    translate = int(translate)
    num_samples = int(batch_size/4)
    
    all_files=glob.glob(os.path.join(file_loc, '*'))
    if int(magnification)!=0:
        all_files = [loc for loc in all_files if loc.rsplit('/', 1)[1].rsplit('-', 1)[0].rsplit('-', 1)[1] == str(magnification)]
    num_batches = int(np.floor(len(all_files)/batch_size))

    while 1:
        random.shuffle(all_files) # randomize after every epoch
        # now get the right files for the batch:
        for index, label in enumerate(label_list):
            x=[]
            y=[]
            batch_files = []
            # get imgs from the current class
            img_locs = [loc for loc in all_files if loc.rsplit('/', 1)[1].split('_', 1)[1].split('-', 1)[0] == str(label)]
            batch_files.append(random.choice(img_locs))

            # get imgs from the same sub class, but not duplicates:
            pi_pos = [x for x in img_locs if x not in batch_files]
            batch_files.append(random.choice(pi_pos))

            # get imgs from the same intra class, but not duplicates:
            pi_neg = [loc for loc in all_files if loc.rsplit('/', 1)[1].split('_', 1)[1].split('-', 1)[0][0] == str(label)[0]]
            pi_neg = [x for x in pi_neg if x not in batch_files]
            batch_files.append(random.choice(pi_neg))

            # get imgs from the same intra class, but not duplicates:
            n_ = [loc for loc in all_files if loc.rsplit('/', 1)[1].split('_', 1)[1].split('-', 1)[0][0] != str(label)[0]]
            batch_files.append(random.choice(n_))

            # Now get some images randomly from the dataset. This is samples 4: batch_size
            for i in range(batch_size-4):
                batch_files.append(random.choice(all_files))

            # now get the images and augment:
            for image_loc in batch_files:
                image = Image.open(image_loc)

                # APPLY AUGMENTATION:
                # flips
                if flips:
                    flip_vert = random.randint(0, 1)
                    flip_hor = random.randint(0, 1)
                    if flip_vert:
                        image = image.transpose(Image.FLIP_TOP_BOTTOM)
                    if flip_hor:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)

                # rotation
                if rotate:
                    square_rot =  bool((np.random.uniform(0, 1, 1)<square_rot_p))
                    if square_rot:  # maybe this is dumb, but it cant hurt
                        angle = random.randint(0, 4)
                        if(angle ==0):
                            image = image.transpose(Image.ROTATE_90)
                        elif(angle ==1):
                            image = image.transpose(Image.ROTATE_180)
                        elif(angle ==2):
                            image = image.transpose(Image.ROTATE_270)
                    else:
                        angle = np.random.uniform(0, 360,1)
                        image=image.rotate(angle)

                if(im_size != 0):
                    image_shape = (im_size, im_size)
                    image = image.resize(image_shape)

                # translate
                ts_sz_row = randint(-1*translate, translate)
                ts_sz_col = randint(-1*translate, translate)

                image = image.transform(image.size, Image.AFFINE, (1, 0, ts_sz_row, 0, 1, ts_sz_col))

                image = np.reshape(np.array(image.getdata()), (im_size, im_size, 3))
                image = image/255.0 
                
                label = image_loc.rsplit('/', 1)[1].split('_', 1)[1].split('-', 1)[0]
                y_temp = label_list.index(label)
                y_temp = np.eye(8)[y_temp]
                
                x.append(image)
                y.append(y_temp)
            x=np.array(x)
            y=np.array(y)
            yield (x, y)

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
from keras.layers import Dropout, Flatten, Reshape, Input, Dense, GlobalAveragePooling2D
from keras.layers.core import Activation, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.engine.topology import Layer
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD 


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


####### Fine tune a model

def ft_pre_model(base_model, data_loc, out_loc, magnification, num_out, epochs_first=10, img_dim=512, layer_train=249):   
    # Params
    epochs_first=int(epochs_first)
    batch_size=16   # make this divisible by len(x_data)
    img_dim=int(img_dim)
    layer_train = int(layer_train)
    
    # Paths to data
    if not os.path.exists(out_loc):
        os.makedirs(out_loc)
    train_loc = os.path.join(str(data_loc), str(magnification), 'train')
    valid_loc = os.path.join(str(data_loc), str(magnification), 'valid')
    num_train = len(glob.glob(train_loc + '/**/*.png', recursive=True))
    num_valid = len(glob.glob(valid_loc + '/**/*.png', recursive=True))
    print('num_train', num_train)
    print('num_valid', num_valid)

    # Set the number of steps per epoch
    steps_per_epoch = np.floor(num_train/batch_size) # num of batches from generator at each epoch. (make it full train set)
    validation_steps = np.floor(num_valid/batch_size)# size of validation dataset divided by batch size
    print('steps_per_epoch', steps_per_epoch)
    print('validation_steps', validation_steps)

    # Image generators
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        shear_range=0.2,
        zoom_range=0.1,
        height_shift_range=.2,
        width_shift_range=.2,
        rotation_range=360,
        horizontal_flip=True,
        vertical_flip=True)

    generator = datagen.flow_from_directory(
            train_loc,
            target_size=(img_dim, img_dim),
            batch_size=batch_size,
            class_mode='categorical')

    valid_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input)

    valid_generator = valid_datagen.flow_from_directory(
            valid_loc,
            target_size=(img_dim, img_dim),
            batch_size=batch_size,
            class_mode='categorical')

    # train the last layers first. We don't need this to be perfect, just get resonable weights before fine tuning
    # add a global spatial average pooling layer for visualization
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_out, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    print('Training the top layers only')
    hist = model.fit_generator(generator,
                                  validation_data=valid_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs_first,
                                  validation_steps=validation_steps)

    print('Make sure nothing is going terribly wrong with training last layers')
    print('Final 2 Epochs Avg Validation loss: ', np.mean(hist.history['val_acc'][-2:]))


    plt.plot(hist.history['loss'])    
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Training top layers')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Train acc', 'val_acc'], loc='upper left')
    plt.figure(figsize=(10,6))
    plt.show()
    
    # at this point, the top layers are well trained and we can start fine-tuning convolutional layers 
    # We will freeze the bottom N layers and train the remaining top layers.
    for layer in model.layers[:layer_train]:
       layer.trainable = False
    for layer in model.layers[layer_train:]:
       layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    print('Fine tuning layers')
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0),
                 ModelCheckpoint(filepath=os.path.join(out_loc, 'fine_tune'+'_.{epoch:02d}-{val_acc:.2f}.hdf5'), 
                                 verbose=1, monitor='val_loss', save_best_only=True)]

    hist = model.fit_generator(generator,
                                  validation_data=valid_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=100,
                                  validation_steps=validation_steps,
                                  callbacks=callbacks)

    print('Check fine-tuning')
    print('Final 5 Epochs Avg Validation loss: ', np.mean(hist.history['val_acc'][-5:]))
    plt.plot(hist.history['loss'])    
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Fine tuning model')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Train acc', 'val_acc'], loc='upper left')
    plt.figure(figsize=(10,6))
    plt.show()


############################



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





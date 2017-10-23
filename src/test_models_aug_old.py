from models import*
import sys

def main(data_loc, out_loc, epochs, batch_size, im_size, model_str):
    import os
    import glob
    import random
    import numpy as np 
    import pandas as pd
    import keras
    import pickle
    from keras import backend as K
    from keras.engine.topology import Layer
    from keras.layers import Dropout, Flatten, Reshape, Input
    from keras.layers.core import Activation, Dense, Lambda
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    from keras.preprocessing.image import ImageDataGenerator


    # Get the function:
    functionList = {
    'conv_6L': conv_6L,
    'vgg16_ft': vgg16_ft,
    'InceptionV3_ft' : InceptionV3_ft
    }

    parameters = {
    'learning_rate': .001,
    'dropout': .1,
    'num_output' : 8,
    'im_size' : im_size
    }

    # Locations
    train_loc = os.path.join(str(data_loc),'train')
    valid_loc = os.path.join(str(data_loc),'valid')
    num_train = len(glob.glob(train_loc + '/**/*.png', recursive=True))
    num_valid = len(glob.glob(valid_loc + '/**/*.png', recursive=True))
    print('train_loc', train_loc)
    print('valid_loc', valid_loc)

    print('num_train', num_train)
    print('num_valid', num_valid)


    # Params for all models
    epochs=int(epochs)
    batch_size=int(batch_size)   # make this divisible by len(x_data)
    steps_per_epoch = np.floor(num_train/batch_size) # num of batches from generator at each epoch. (make it full train set)
    validation_steps = np.floor(num_valid/batch_size)# size of validation dataset divided by batch size
    print('steps_per_epoch', steps_per_epoch)
    print('validation_steps', validation_steps)
    print('batch_size', batch_size)

    image_shape = (int(im_size), int(im_size))

    train_datagen = ImageDataGenerator(
        rescale=1./255)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_loc,
        target_size=image_shape,
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            valid_loc,
            target_size=image_shape,
            batch_size=batch_size,
            class_mode='categorical')

    
    model = functionList[model_str](**parameters)
    print(model_str)
    print(model.summary())
    name = model_str+'noaug'
    out_file=os.path.join(str(out_loc), name)
    callbacks = [EarlyStopping(monitor='val_loss', patience=15, verbose=0),
    keras.callbacks.LearningRateScheduler(triangular2), 
    ModelCheckpoint(filepath=os.path.join(out_loc, name+'_.{epoch:02d}-{val_acc:.2f}.hdf5'), verbose=1, monitor='val_loss', save_best_only=True)]

    hist = model.fit_generator(train_generator,
                                      validation_data=validation_generator,
                                      steps_per_epoch=steps_per_epoch, 
                                      epochs=epochs,
                                      validation_steps=validation_steps,
                                      callbacks=callbacks)

    pickle.dump(hist.history, open(out_file, 'wb'))


if __name__ == "__main__":
    data_loc = sys.argv[1]
    out_loc = sys.argv[2]
    epochs = sys.argv[3]
    batch_size = sys.argv[4]
    im_size = sys.argv[5]
    model_str = sys.argv[6]

    main(data_loc, out_loc, epochs, batch_size, im_size, model_str)
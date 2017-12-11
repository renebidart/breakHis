from models import*
import sys

def main(data_loc, out_loc, epochs, batch_size, magnification, img_dim, num_output):
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
    from keras.applications.vgg16 import VGG16, preprocess_input


    # Params

    epochs=int(epochs)
    batch_size=int(batch_size)
    img_dim=int(img_dim)

    # Locations
    if not os.path.exists(out_loc):
        os.makedirs(out_loc)
    train_loc = os.path.join(str(data_loc), str(magnification), 'train')
    valid_loc = os.path.join(str(data_loc), str(magnification), 'valid')
    num_train = len(glob.glob(train_loc + '/**/*.png', recursive=True))
    num_valid = len(glob.glob(valid_loc + '/**/*.png', recursive=True))
    print('num_train', num_train)
    print('num_valid', num_valid)

    steps_per_epoch = np.floor(num_train/batch_size) # num of batches from generator at each epoch. (make it full train set)
    validation_steps = np.floor(num_valid/batch_size)# size of validation dataset divided by batch size
    print('steps_per_epoch', steps_per_epoch)
    print('validation_steps', validation_steps)
    print('batch_size', batch_size)

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
            target_size=(img_dim, img_dim), # (704, 448)  (700X460)-true size
            batch_size=batch_size,
            class_mode='categorical')

    valid_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input)

    valid_generator = valid_datagen.flow_from_directory(
            valid_loc,
            target_size=(img_dim, img_dim),
            batch_size=batch_size,
            class_mode='categorical')


    # loads the model with image net weights and only the dense layer after global average pooling set to trainable
    # weights_path = '/home/rbbidart/project/rbbidart/models/vgg16_weights.h5'
    weights_path = '/home/rbbidart/breakHis_out/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    model = get_vgg_var_sz(weights_path, learning_rate=.001, dropout = .1, num_output=8, img_dim=img_dim)
    print(model.summary())
    name = 'vgg_pretrained_vis_'+magnification
    out_file=os.path.join(str(out_loc), name)
    callbacks = [EarlyStopping(monitor='val_loss', patience=15, verbose=0),
    ModelCheckpoint(filepath=os.path.join(out_loc, name+'_.{epoch:02d}-{val_acc:.2f}.hdf5'), verbose=1, monitor='val_loss', save_best_only=True)]

    hist = model.fit_generator(generator,
                                  validation_data=valid_generator,
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
    magnification = sys.argv[5]
    img_dim = sys.argv[6]
    num_output = sys.argv[7]


    main(data_loc, out_loc, epochs, batch_size, magnification, img_dim, num_output)


from models import*
import sys

def main(data_loc, out_loc, magnification):
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

    size = int(magnification)
    samples = 8

    start = 1
    n_folds = 6
    def get_freatures_vgg(generator, loc, samples=8, classes=8, batch_size=1):
        num_imgs = sum([len(files) for r, d, files in os.walk(loc)])
        num_samples = samples*num_imgs
        print('num_imgs', num_imgs)
        print('num_samples', num_samples)

        from keras.applications.vgg16 import VGG16
        base_model = VGG16(weights='imagenet')
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

        all_features = np.zeros((num_samples, 4096+classes))

        for i in range(0, num_samples, batch_size):
            x, y = next(generator)
            features = model.predict(x)
    #         all_features[i:i+len(features), 0:classes] = y
    #         all_features[i:i+len(features), classes:] = features
            all_features[i, 0:classes] = y
            all_features[i, classes:] = features
        print('np.sum(all_features[:, :8]', np.sum(all_features[:, :8]))
        print('all_features.shape', all_features.shape)
        return all_features


    for i in range(1, n_folds, 1):
        new_dir = 'train'
        fold = 'fold'+str(i)
        cur_loc = os.path.join(data_loc, fold, str(size), new_dir)
        print(cur_loc)

        new_loc = os.path.join(out_loc, fold, str(size),  new_dir)
        if not os.path.exists(new_loc):
            os.makedirs(new_loc)
            
        new_loc = os.path.join(out_loc, fold, str(size),  new_dir)

        datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        shear_range=0.2,
        zoom_range=0.2,
        height_shift_range=.2,
        width_shift_range=.2,
        rotation_range=360,
        horizontal_flip=True,
        vertical_flip=True)

        generator = datagen.flow_from_directory(
                cur_loc,
                target_size=(224, 224),
                batch_size=1,
                class_mode='categorical')

        all_features = get_freatures_vgg(generator, cur_loc, samples=samples, classes=8, batch_size=1)
        np.save(os.path.join(new_loc, new_dir+'_feat_vgg_'+str(size)+'_aug1.npy'), all_features)

        
    for i in range(1, n_folds, 1):
        new_dir = 'valid'
        fold = 'fold'+str(i)
        cur_loc = os.path.join(data_loc, fold, str(size), new_dir)
        print(cur_loc)

        new_loc = os.path.join(out_loc, fold, str(size),  new_dir)
        if not os.path.exists(new_loc):
            os.makedirs(new_loc)
        new_loc = os.path.join(out_loc, fold, str(size),  new_dir)

        datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        shear_range=0.2,
        zoom_range=0.2,
        height_shift_range=.2,
        width_shift_range=.2,
        rotation_range=360,
        horizontal_flip=True,
        vertical_flip=True)

        generator = datagen.flow_from_directory(
                cur_loc,
                target_size=(224, 224),
                batch_size=1,
                class_mode='categorical')

        all_features = get_freatures_vgg(generator, cur_loc, samples=samples, classes=8, batch_size=1)
        np.save(os.path.join(new_loc, new_dir+'_feat_vgg_'+str(size)+'_aug1.npy'), all_features)
        
        
    for i in range(start, n_folds, 1):
        new_dir = 'test'
        fold = 'fold'+str(i)
        cur_loc = os.path.join(data_loc, fold, str(size), new_dir)
        print(cur_loc)

        new_loc = os.path.join(out_loc, fold, str(size),  new_dir)
        if not os.path.exists(new_loc):
            os.makedirs(new_loc)

        datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input)

        generator = datagen.flow_from_directory(
                cur_loc,
                target_size=(224, 224),
                batch_size=1,
                class_mode='categorical')

        all_features = get_freatures_vgg(generator, cur_loc, samples=1, classes=8, batch_size=1)
        np.save(os.path.join(new_loc, new_dir+'_feat_vgg_'+str(size)+'_aug1.npy'), all_features)


if __name__ == "__main__":
    data_loc = sys.argv[1]
    out_loc = sys.argv[2]
    magnification = sys.argv[3]

    main(data_loc, out_loc, magnification)
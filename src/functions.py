import os
import sys
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
from shutil import copyfile

def get_freatures_vgg(generator, loc, samples=8, classes=8, batch_size=1):
    num_imgs = sum([len(files) for r, d, files in os.walk(loc)])
    num_samples = samples*num_imgs
    print('num_imgs', num_imgs)
    print('num_samples', num_samples)

    from keras.applications.vgg16 import VGG16
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
    print('loaded VGG')

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


# ?
# def get_freatures_incp3(generator, loc, samples=8, classes=8, batch_size=32):
#     num_imgs = sum([len(files) for r, d, files in os.walk(loc)])
#     num_samples = samples*num_imgs

#     from keras.applications.inception_v3 import InceptionV3
#     base_model = InceptionV3(weights='imagenet', include_top=False)
#     x = base_model.output
#     features = GlobalAveragePooling2D()(x)
#     model = Model(outputs=features, inputs=base_model.input)


#     all_features = np.zeros((num_samples, 2048+classes))

#     for i in range(0, num_samples, batch_size):
#         x, y = next(generator)
#         features = model.predict(x)
#         all_features[i:i+len(y), 0:classes] = y
#         all_features[i:i+len(features), classes:] = features
#     return all_features
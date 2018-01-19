import os
import sys
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
from shutil import copyfile
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imshow
from IPython.core.display import display

from keras.models import Model



def cv_features(model_2, model_8, base_data_dir):
    import itertools
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix

    results_2 = pd.DataFrame(index=range(5), columns=['test_acc', 'train_acc', 'fold'])
    results_8 = pd.DataFrame(index=range(5), columns=['test_acc', 'train_acc', 'fold'])

    for i in range(5):
        fold = 'fold'+str(i+1)
        print('Fold ', i )
        # load the train data
        train_feat_loc=os.path.join(base_data_dir, 'features', 'vgg', fold ,'100','train','train_feat_vgg_100_aug1.npy')
        valid_feat_loc=os.path.join(base_data_dir, 'features', 'vgg', fold ,'100','valid','valid_feat_vgg_100_aug1.npy')

        train_features = np.load(train_feat_loc)
        valid_features = np.load(valid_feat_loc)
        train_features = np.concatenate((train_features, valid_features), axis=0)
        y_train = train_features[: ,:8]
        x_train = train_features[: ,8:]

        # Make y-values indicating benign vs. malignant
        y_bin_train = np.zeros((y_train.shape[0],2))
        for index, row in enumerate(y_train):
            if(np.sum(row[:4])>0):
                y_bin_train[index, :] = 0
                y_bin_train[index, 0] = 1
            else:
                y_bin_train[index, :] = 0
                y_bin_train[index, 1] = 1

        # load the test data
        test_feat_loc=os.path.join(base_data_dir, 'features', 'vgg', fold ,'100','test','test_feat_vgg_100_aug1.npy')
        test_features = np.load(test_feat_loc)
        y_test = test_features[:,:8]
        x_test = test_features[:,8:]

        # Make y-values indicating benign vs. malignant
        y_bin_test = np.zeros((y_test.shape[0],2))
        for index, row in enumerate(y_test):
            if(np.sum(row[:4])>0):
                y_bin_test[index, :] = 0
                y_bin_test[index, 0] = 1
            else:
                y_bin_test[index, :] = 0
                y_bin_test[index, 1] = 1

        # get rid of hot-one
        y_bin_train = np.argmax(y_bin_train, axis=1)
        y_bin_test = np.argmax(y_bin_test, axis=1)
        y_8_train = np.argmax(y_train, axis=1)
        y_8_test = np.argmax(y_test, axis=1)


        ########### 2 CLASS  ###################
        # Fit the model:
        clf = model_2
        clf.fit(x_train, y_bin_train)

        #train and test acc
        y_pred = clf.predict(x_train)
        train_acc = accuracy_score(y_bin_train, y_pred)
        y_pred_2 = clf.predict(x_test)
        test_acc=accuracy_score(y_bin_test, y_pred_2)
        results_2.loc[i] = [test_acc, train_acc, fold]


        ########### 8 CLASS  ###################
        # Fit the model:
        clf = model_8
        clf.fit(x_train, y_8_train)

        #train and test acc
        y_pred = clf.predict(x_train)
        train_acc = accuracy_score(y_8_train, y_pred)
        y_pred_8 = clf.predict(x_test)
        test_acc=accuracy_score(y_8_test, y_pred_8)
        results_8.loc[i] = [test_acc, train_acc, fold]


    print('Binary classification results:')
    display(results_2)
    print('Average Test Accc: ', results_2["test_acc"].mean())

    print('Binary Classification Confusion matrix for fold 5:')
    cm = confusion_matrix(y_bin_test, y_pred_2)
    classes = ['B', 'M']

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # add the numbers
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    
    print('8-Class Classification Results:')
    display(results_8)
    print('Average Test Accc: ', results_8["test_acc"].mean())

    print('8-Class Classification Confusion matrix for fold 5:')
    cm = confusion_matrix(y_8_test, y_pred_8)
    classes = ['B_A', 'B_F', 'B_PT', 'B_TA', 'M_DC', 'M_LC', 'M_MC', 'M_PC']

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # add the numbers
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


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


def get_freatures_incp3(model, loc, classes=8, image_shape = (512, 512)):
    all_imgs = glob.glob(loc+'/**/*.png', recursive=True)
    num_samples = len(all_imgs)

    conv_model = Model(inputs=model.input, outputs=model.get_layer(index=311).output)
    all_features = np.zeros((num_samples, 2048))
    features_names = []

    for i, image_loc in enumerate(all_imgs):
        image = Image.open(image_loc)
        image = np.array(image.resize(image_shape))
        features = conv_model.predict(np.expand_dims(image, axis=0))

        features_names.append(image_loc.rsplit('/')[-1])
        all_features[i, :] = features
    return features_names, all_features
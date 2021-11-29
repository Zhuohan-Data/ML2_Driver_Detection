# -*- coding: utf-8 -*-
#%%
import keras.applications.vgg16

import numpy as np
import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
                                       ZeroPadding2D

# from keras.layers.normalization import BatchNormalization
# from keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
# from sklearn.metrics import log_loss
from numpy.random import permutation
import tensorflow as tf
from tensorflow import keras
import random
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import to_categorical
#%%
# os.chdir('/home/ubuntu/ML/Driver')

## Process images in parallel
AUTOTUNE = tf.data.AUTOTUNE
# os.chdir("..")
# DATA_DIR = os.getcwd()  + os.path.sep +'imgs' + os.path.sep
DATA_DIR = os.getcwd()+os.path.sep+'data' + os.path.sep +'imgs' + os.path.sep
sep = os.path.sep

n_epoch = 2
BATCH_SIZE = 16

CHANNELS = 3
IMAGE_SIZE = 100
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_uniform(seed=SEED)
DROPOUT = 0.2
LR = 1e-3


# color type: 1 - grey, 3 - rgb
color_type_global = 3

# color_type = 1 - gray
# color_type = 3 - RGB

# os.chdir('/home/ubuntu/ML/Driver')
#os.chdir("/home/ubuntu/ML2/Project/")

def process_target(target_type):

    dict_target = {}
    xerror = 0

    if target_type == 1:
        final_target = to_categorical(list(ds_targets))
    else:
        # final_target = list(xdf_dset['target'])
        final_target = list(ds_targets)
    return final_target


def process_path(feature, target):
    # Processing Label

    label = target
    # Processing feature
    # load the raw data from the file as a string
    file_path = feature
    img = tf.io.read_file(file_path)

    img = tf.io.decode_jpeg(img, channels=CHANNELS)

    resized = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    return resized

def read_data(target_type,split='train'):
    ## Only the training set
    ## xdf_dset ( data set )
    ## read the data data from the file
    if split=='train':
        ds_inputs = np.array(DATA_DIR + 'train/' + xdf_dset['classname'] + '/' + xdf_dset['img'])
    if split == 'test':
        test_path = os.path.join(DATA_DIR, 'test', '*.jpg')
        ds_inputs = glob.glob(test_path)

    print(ds_targets)
    ds_targets = process_target(target_type)

    ## Make the channel as a list to make it variable
    list_ds = tf.data.Dataset.from_tensor_slices((ds_inputs,ds_targets))

    final_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

    return final_ds


#%%
def save_model(model, index, cross=''):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)


def read_model(index, cross=''):
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    model = model_from_json(open(os.path.join('cache', json_name)).read())
    model.load_weights(os.path.join('cache', weight_name))
    return model


def split_validation_set(train, target, test_size):
    random_state = 51
    X_train, X_test, y_train, y_test = \
        train_test_split(train, target,
                         test_size=test_size,
                         random_state=random_state)
    return X_train, X_test, y_train, y_test


def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3',
                                                 'c4', 'c5', 'c6', 'c7',
                                                 'c8', 'c9'])
    #result1.loc['img',:] = pd.Series(test_id, index=result1.index)
    result1.insert(loc=0, column='img', value=pd.Series(test_id, index=result1.index))
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)


def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def merge_several_folds_geom(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a *= np.array(data[i])
    a = np.power(a, 1/nfolds)
    return a.tolist()


def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
    data = []
    target = []
    index = []
    for i in range(len(driver_id)):
        if driver_id[i] in driver_list:
            data.append(train_data[i])
            target.append(train_target[i])
            index.append(i)
    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    index = np.array(index, dtype=np.uint32)
    return data, target, index


def definite_model(img_rows, img_cols, color_type=1):
    # Add the pretrained layers
    pretrained_model = keras.applications.ResNet50(include_top=False, weights='imagenet')

    # Add GlobalAveragePooling2D layer
    dropout = keras.layers.Dropout(rate=0.5)(pretrained_model.output)

    # Add GlobalAveragePooling2D layer
    average_pooling = keras.layers.GlobalAveragePooling2D()(dropout)

    # Add the output layer
    output = keras.layers.Dense(10, activation='softmax')(average_pooling)

    # Get the model
    model = keras.Model(inputs=pretrained_model.input, outputs=output)


    # Learning rate is changed to 0.001
    sgd = SGD(learning_rate=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model


def run_cross_validation(nfolds=10, nb_epoch=10, split=0.2, modelStr=''):

    # Now it loads color image
    # input image dimensions
    img_rows, img_cols = 100, 100
    batch_size = 64
    random_state = 20

    train_data = read_data(target_type=1, split='train')

    # ishuf_train_data = []
    # shuf_train_target = []
    # index_shuf = range(len(train_target))
    # shuffle(index_shuf)
    # for i in index_shuf:
    #     shuf_train_data.append(train_data[i])
    #     shuf_train_target.append(train_target[i])

    # yfull_train = dict()
    # yfull_test = []
    num_fold = 0
    # kf = KFold(n_splits=nfolds,
    #            shuffle=True, random_state=random_state)
    # for  train_drivers, test_drivers in kf.split(unique_drivers):
    num_fold += 1
    print('Start KFold number {} from {}'.format(num_fold, nfolds))
    # print('Split train: ', len(X_train), len(Y_train))
    # print('Split valid: ', len(X_valid), len(Y_valid))
    # print('Train drivers: ', unique_list_train)
    # print('Test drivers: ', unique_list_valid)
    # model = create_model_v1(img_rows, img_cols, color_type_global)
    # model = vgg_bn_model(img_rows, img_cols, color_type_global)
    model = definite_model(img_rows, img_cols, color_type_global)

    model.fit(train_data, batch_size=batch_size,
              epochs=nb_epoch, verbose=1,
              validation_split=split, shuffle=True)
        #print('losses: ' + hist.history.losses[-1])

        #print('Score log_loss: ', score[0])

    save_model(model, num_fold, modelStr)



def test_model_and_submit(start=1, end=1, modelStr=''):
    img_rows, img_cols = 50, 50
    # batch_size = 64
    # random_state = 51
    nb_epoch = 15

    print('Start testing............')
    test_data = read_data(target_type=1, split='train')

    test_path = os.path.join(DATA_DIR, 'test', '*.jpg')
    images = glob.glob(test_path)

    test_ids = []
    for img_path in images:
        img_id = os.path.basename(img_path)
        test_ids.append(img_id)


    yfull_test = []

    for index in range(start, end + 1):
        # Store test predictions
        model = read_model(index, modelStr)
        test_prediction = model.predict(test_data, batch_size=32, verbose=1)

        yfull_test.append(test_prediction)

    info_string = 'loss_' + modelStr \
                  + '_r_' + str(img_rows) \
                  + '_c_' + str(img_cols) \
                  + '_folds_' + str(end - start + 1) \
                  + '_ep_' + str(nb_epoch)

    test_res = merge_several_folds_mean(yfull_test, end - start + 1)
    create_submission(test_res, test_id, info_string)
#%%
# FILE_NAME = os.getcwd()+ os.path.sep+  'driver_imgs_list.csv'
FILE_NAME = os.getcwd()+os.path.sep+'data'+ os.path.sep+  'driver_imgs_list.csv'
xdf_dset = pd.read_csv(FILE_NAME)
class_names = np.sort(xdf_dset['classname'].unique())
x = lambda x : tf.argmax(x ==  class_names).numpy()
split_list = []
for i in range(len(xdf_dset)):
    split_list.append('train') if xdf_dset['subject'][i] == 'p002' else split_list.append('validation')
xdf_dset['split'] = split_list
xdf_dset['target'] = xdf_dset['classname'].apply(x)
# xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()

driver_id = xdf_dset['subject']
unique_drivers = sorted(list(set(driver_id)))
print('Unique drivers: {}'.format(len(unique_drivers)))
print(unique_drivers)

#%%
# nfolds, nb_epoch, split
run_cross_validation(2, 3, 0.15, 'Resnet50')

model.summary()

# nb_epoch, split
# run_one_fold_cross_validation(10, 0.1)
#%%
test_model_and_submit(1, 1, 'Resnet50')


#%%
read_data(target_type=1, split='train')
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

#import tensorflow_addons as tfa
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
# os.chdir("/home/ubuntu/ML2/Project/")

## Process images in parallel
AUTOTUNE = tf.data.AUTOTUNE
# os.chdir('/home/ubuntu/ML/Driver')
# DATA_DIR = os.getcwd()  + os.path.sep +'imgs' + os.path.sep
# CSV_DIR = os.getcwd()+os.path.sep+'driver_imgs_list.csv'
DATA_DIR = os.getcwd()+os.path.sep+'data' + os.path.sep +'imgs' + os.path.sep
CSV_DIR = os.getcwd()+os.path.sep+'data'+os.path.sep+'driver_imgs_list.csv'
sep = os.path.sep

nfolds=5
n_epoch = 5
BATCH_SIZE = 8

CHANNELS = 3
IMAGE_SIZE = 224

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_uniform(seed=SEED)
random_state = 51

DROPOUT = 0.2
LR = 1e-3


# color type: 1 - grey, 3 - rgb
color_type_global = 3

# color_type = 1 - gray
# color_type = 3 - RGB

modelnames=['Resnet50','Resnet50']

def process_target(target_type,target):


    if target_type == 1:
        final_target = to_categorical(list(target))
    else:
        final_target = list(target)
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
    return resized,label

def process_path_test(feature):
    # Processing Label

    # label = target
    # Processing feature
    # load the raw data from the file as a string
    file_path = feature
    img = tf.io.read_file(file_path)

    img = tf.io.decode_jpeg(img, channels=CHANNELS)

    resized = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    return resized

def read_data(target_type,split='train',train_index=0,valid_index=0,method='split'):

    if split=='train':

        if method=='split':
            train_data,val_data,train_targets,val_targets =split_validation_set(ds_inputs,ds_targets,0.2)
        else:
            train_data, val_data = ds_inputs[train_index], ds_inputs[valid_index]
            train_targets, val_targets = ds_targets[train_index], ds_targets[valid_index]
        train_targets = process_target(target_type, train_targets)
        val_targets = process_target(target_type, val_targets)
        train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_targets))
        val_ds = tf.data.Dataset.from_tensor_slices((val_data, val_targets))
        final_train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
        final_val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

        return final_train_ds, final_val_ds

    if split == 'test':
        test_path = glob.glob(os.path.join(DATA_DIR, 'test', '*.jpg'))
        test_data = tf.data.Dataset.from_tensor_slices(test_path)
        final_test_ds = test_data.map(process_path_test, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
        return final_test_ds


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

#%%
def test_ensemble_and_submit(start=1, end=1, modelStr='',modelnames=[1,2],method='split'):
    img_rows, img_cols = IMAGE_SIZE,IMAGE_SIZE
    # batch_size = 64
    # random_state = 51

    print('Start testing............')
    test_data = read_data(target_type=1, split='test')

    test_path = os.path.join(DATA_DIR, 'test', '*.jpg')
    images = glob.glob(test_path)

    test_ids = []
    for img_path in images:
        img_id = os.path.basename(img_path)
        test_ids.append(img_id)


    yfull_test = []
    if method=='split':
        model1 = read_model(1,modelnames[0])
        test_prediction = model1.predict(test_data, verbose=1)

        yfull_test.append(test_prediction)

        model2 = read_model(1,modelnames[1])
        test_prediction = model2.predict(test_data, verbose=1)
        yfull_test.append(test_prediction)

        test_res = merge_several_folds_mean(yfull_test, 2)
    else:
        for i in modelnames:
            for index in range(start, end + 1):
                # Store test predictions
                model = read_model(index, i)
                test_prediction = model.predict(test_data,verbose=1)

                yfull_test.append(test_prediction)

        test_res = merge_several_folds_mean(yfull_test, (end - start + 1)*len(modelnames))
    info_string = 'loss_' + modelStr \
                  + '_r_' + str(img_rows) \
                  + '_c_' + str(img_cols) \
                  + '_folds_' + str(end - start + 1)
    # model = read_model('best',modelStr)
    # test_res = model.predict(test_data)
    create_submission(test_res, test_ids, info_string)
    print('finished')
#%%
xdf_dset = pd.read_csv(CSV_DIR)
class_names = np.sort(xdf_dset['classname'].unique())
x = lambda x : tf.argmax(x ==  class_names).numpy()
# split_list = []
# for i in range(len(xdf_dset)):
#     split_list.append('train') if xdf_dset['subject'][i] == 'p002' else split_list.append('validation')
# xdf_dset['split'] = split_list
xdf_dset['target'] = xdf_dset['classname'].apply(x)
# xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()

# driver_id = xdf_dset['subject']
# unique_drivers = sorted(list(set(driver_id)))
# print('Unique drivers: {}'.format(len(unique_drivers)))
# print(unique_drivers)
#%%
ds_inputs = np.array(DATA_DIR + 'train/' + xdf_dset['classname'] + '/' + xdf_dset['img'])
ds_targets = xdf_dset['target']
#%%
# nfolds, nb_epoch, split
# run_cross_validation(2, 5, 0.15, 'Resnet50')
# run_cross_validation(nfolds,n_epoch,0.15,"Resnet50",method='split')
# run_ensemble(nfolds,n_epoch,0.15,"Ensemble2Resnet50",method='split',modelnames=modelnames)
# model.summary()

# nb_epoch, split
# run_one_fold_cross_validation(10, 0.1)
#%%
# test_model_and_submit(0,0, 'Resnet50')
test_ensemble_and_submit(1,nfolds,'Resnet50',modelnames=modelnames,method='split')
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
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import sys
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score,  matthews_corrcoef
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
#%%
## Process images in parallel
AUTOTUNE = tf.data.AUTOTUNE
# os.chdir("..")
DATA_DIR = os.getcwd() + os.path.sep + 'data'+os.path.sep +'imgs' + os.path.sep
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
#%%
def process_target(target_type,ds_targets):
    '''
    :return:
    '''

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

    img = tf.image.resize( img, [IMAGE_SIZE, IMAGE_SIZE])
    # Data augmentation
    # seed = (1,0)
    # img = (img/255.0)
    # img = tf.image.rot90(img)
    # img = tf.image.resize_with_crop_or_pad(img, IMAGE_SIZE + 6, IMAGE_SIZE + 6)
    # img = tf.image.stateless_random_crop(
    #     img, size=[IMAGE_SIZE, IMAGE_SIZE, CHANNELS],seed=seed)
    #
    # img = tf.clip_by_value(img, 0, 1)
    # Reshape
    # img = tf.reshape(img, [-1])

    return img, label

def read_data(target_type,split='train'):
    ## Only the training set
    ## xdf_dset ( data set )
    ## read the data data from the file
    if split=='train':
        ds_inputs = np.array([])
        ds_targets = np.array([])
        k = np.array([])
        for j in range(10):
            path = os.path.join(DATA_DIR, 'train',
                                'c'+str(j), '*.jpg')
            ds_inputs=np.append(ds_inputs,glob.glob(path))
            k = np.append(k,len(ds_inputs)-k.sum())
            for i in range(int(k[j])):
                ds_targets = np.append(ds_targets,j)
    else:
        # ds_inputs = np.array(DATA_DIR + 'test' + xdf_dset['classname'] + xdf_dset['img'])
        print('not now')
    ds_targets = process_target(target_type,ds_targets)

    ## Make the channel as a list to make it variable


    list_ds = tf.data.Dataset.from_tensor_slices((ds_inputs,ds_targets))


    final_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
    return final_ds

def definite_model():
    # Add the pretrained layers
    pretrained_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet')

    # Add GlobalAveragePooling2D layer
    dropout = keras.layers.Dropout(rate=0.5)(pretrained_model.output)

    # Add GlobalAveragePooling2D layer
    average_pooling = keras.layers.GlobalAveragePooling2D()(dropout)

    # Add the output layer
    output = keras.layers.Dense(10, activation='softmax')(average_pooling)

    # Get the model
    model = keras.Model(inputs=pretrained_model.input, outputs=output)

    model.summary()

    # Learning rate is changed to 0.001
    sgd = SGD(learning_rate=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model

def train_func(train_ds):
    # Use a breakpoint in the code line below to debug your script.

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience =3)
    check_point = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='accuracy', save_best_only=True)

    final_model = definite_model()

    final_model.fit(train_ds,  epochs=n_epoch, callbacks=[early_stop, check_point],
                    )
#%%
FILE_NAME = os.getcwd()+ os.path.sep+ "data" + os.path.sep + 'driver_imgs_list.csv'
xdf_dset = pd.read_csv(FILE_NAME)
class_names = np.sort(xdf_dset['classname'].unique())
x = lambda x : tf.argmax(x ==  class_names).numpy()

xdf_dset['target'] = xdf_dset['classname'].apply(x)
# xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()
#%%
train_ds = read_data(target_type=1,split='train')
train_func(train_ds)
# -*- coding: utf-8 -*-
###{{{
import numpy as np

import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
import datetime

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
                                       ZeroPadding2D

# from keras.layers.normalization import BatchNormalization
# from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
# from sklearn.metrics import log_loss
from numpy.random import permutation
###}}}

NB_CLASSES = 10
np.random.seed(2016)
use_cache = 1
color_type_global = 3

# color_type = 1 - gray
# color_type = 3 - RGB


def get_im(path, img_rows, img_cols, color_type=1):
###{{{
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized
###}}}

#num_folders is how many folders cn we have
def load_train_as_list(p, num_folders):
###{{{
    X_train = []
    y_train = []
    driver_id = []

    print("...enter load_train_as_list...")
    for j in xrange(num_folders):
        path = os.path.join(p, 'c'+str(j), '*jpg')
        files = glob.glob(path)
        print("number of files: " + str(len(files)))
        for fl in files:
            # print(str(fl))
            flbase = os.path.basename(fl)
            X_train.append(str(fl))
            y_train.append(j)
    #endfor
    print("len X_train: "+str(len(X_train))+". len y_train: "+str(len(y_train)))
    unique_drivers = []
    print("...exit load_train...")
    return X_train, y_train, driver_id, unique_drivers
    pass
###}}}


def load_train(img_rows, img_cols, color_type=1):
###{{{
    X_train = []
    y_train = []
    driver_id = []
    # leave driver_id empty. we don't need it and I'm too lazy to remove and change all functions involving driver_id`
    # deleted driver_data method
    # driver_data = get_driver_data()

    print('Read train images')
    for j in xrange(10):
    # for j in xrange(2):
        print('Load folder c{}'.format(j))#piemel
        #path = os.path.join('..', 'input', 'imgs', 'train',
                            #'c' + str(j), '*.jpg')
        # gabi2 = "/home/ml0501/statefarm/train/"
        gabi2 = "/home/ml0505/train2/"
        # gabi2 = "/home/ml0505/train/"
        # gabi2 = "/home/ml0505/other/"
        path = os.path.join(gabi2,'c' + str(j), '*.jpg')
        # print(str(path))
        files = glob.glob(path)
        print(str(len(files)))
        for fl in files:
            print(str(fl))
            flbase = os.path.basename(fl)
            img = get_im(fl, img_rows, img_cols, color_type)
            X_train.append(img)
            y_train.append(j)
            # driver_id.append(driver_data[flbase]) #piemel
    print("len X_train: "+str(len(X_train))+". len y_train: "+str(len(y_train)))
    unique_drivers = []
    print("...exit load_train...")
    return X_train, y_train, driver_id, unique_drivers
###}}}


def load_test(img_rows, img_cols, color_type=1):
###{{{
    print('Read test images')
    #path = os.path.join('..', 'input', 'imgs', 'test', '*.jpg')
    gabi3 = "/home/ml0501/statefarm/test/"
    path = os.path.join(gabi3,'*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files)/10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im(fl, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))
    print("...load_test...")
    return X_test, X_test_id
###}}}


def cache_data(data, path, i):
###{{{
    print("...enter cache_data...")
    print("i="+str(i))
    print("type data = "+str(type(data))+ " data[0]: "+ str(type(data[0]))+ " data[1]:"+str(type(data[1])))
    # print("data type, shape: "+str(type(data))+" ,"+ str(data[0].shape)+" ,"+str(data[1].shape))

    path2 = path+"file" + str(i) + ".dat"
    # if not os.path.isdir('cache'):
        # os.mkdir('cache')
    if not os.path.isdir(path):
        os.mkdir(path)
    if os.path.isdir(os.path.dirname(path2)):
        file = open(path2, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')
###}}}


def get_imgs_from_list(ilist):
###{{{
    imgs = []
    print("...enter get_imgs_from_list...")
    img_rows, img_cols, color_type = 224,224,3
    i = 0
    print("length ilist: "+str(len(ilist)))
    for item in ilist:
        print(i)
        im = get_im(item, img_rows, img_cols, color_type)
        imgs.append(im)
        i+=1
    #endfor
    imgs_array = np.array(imgs,dtype=np.uint8)
    return imgs_array
    pass
###}}}


def cache_data_from_list(data,path, i, color_type=1):
#data = ([],[])
###{{{
    print("...enter cache_data_from_list...")
    path2 = path+"file" + str(i) + ".dat"
    if not os.path.isdir(path):
        os.mkdir(path)
    if os.path.isdir(os.path.dirname(path2)):
        file = open(path2, 'wb')
        imgs = get_imgs_from_list(data[0])
        trg = np.array(data[1], dtype=np.uint8)
        print("type imgs: "+str(type(imgs))+" type trg: "+str(type(trg)))


#---<3
        if color_type == 1:
            imgs = imgs.reshape(imgs.shape[0], color_type,
                                            img_rows, img_cols)
        else:
            imgs = imgs.transpose((0, 3, 1, 2))

        trg = np_utils.to_categorical(trg, 10)
        trg = trg.astype('float32')
        mean_pixel = [103.939, 116.779, 123.68]
        for c in xrange(3):
            imgs[:, c, :, :] = imgs[:, c, :, :] - mean_pixel[c]
#---<3
        print("type imgs: "+str(type(imgs))+" type trg: "+str(type(trg)))
        pickle.dump((imgs,trg), file)
        file.close()
    else:
        print('Directory doesnt exists')
###}}}
    pass


def restore_data(path):
###{{{
    data = {}
    if os.path.isfile(path):
        print('Restore data from pickle........')
        file = open(path, 'rb')
        data = pickle.load(file)
        print("type data = "+str(type(data)))
    print("...exit restore_data...")
    return data
###}}}


def restore_data2(path):
###{{{
    data = ()
    if os.path.isfile(path):
        print('Restore data from pickle........')
        file = open(path, 'rb')
        data = pickle.load(file)
    print("...exit restore_data...")
    return data
    pass
###}}}


def save_model2(model):
###{{{
    now = str(datetime.date.today())
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')

    json_name = 'architecture'+now+'.json'
    weight_name = 'model_weights'+now+'.h5'
    open(os.path.join('cache_all', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)
    print("...exit save_model2...")
    pass
###}}}


def read_model2():
###{{{
    print("...enter read_model2...")
    json_name = 'architecture.json'
    weight_name = 'model_weights.h5'
    model = model_from_json(open(os.path.join('cache', json_name)).read())
    model.load_weights(os.path.join('cache', weight_name))
    print("...exit read_model2...")
    return model
###}}}


def read_model(index, cross=''):
###{{{
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    model = model_from_json(open(os.path.join('cache', json_name)).read())
    model.load_weights(os.path.join('cache', weight_name))
    print("...exit read_model...")
    return model
###}}}


def create_submission(predictions, test_id, info):
###{{{
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3',
                                                 'c4', 'c5', 'c6', 'c7',
                                                 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm_two'):
        os.mkdir('subm_two')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm_two', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)
    print("...exit create_submission...")
###}}}


def read_and_normalize_and_shuffle_train_data(img_rows, img_cols,
                                              color_type):
###{{{
    # cache_path = os.path.join('/home/ml0502/cache', 'train_r_' + str(img_rows) +
                              # '_c_' + str(img_cols) + '_t_' +
                              # str(color_type) + '.dat')
    # print(cache_path)
    # cache_path = '/home/ml0502/cache/train_r_'+str(img_rows)+'_c_'+str(img_cols)+'_t_'+str(color_type)+'.dat'
    # print(cache_path)

    cache_path_w_file = '/home/ml0502/cache/file.dat'
    cache_path_only_folder = '/home/ml0502/cache/'
    if not os.path.isfile(cache_path_w_file) or use_cache == 0:
    #normal array
        train_data, train_target, driver_id, unique_drivers = \
            load_train(img_rows, img_cols, color_type)
    #pkl
        cache_data((train_data, train_target, driver_id, unique_drivers),
                   cache_path_only_folder, 666)
        print("type train_data, train_target: "+ str(type(train_data))+" "+str(type(train_target)))
    else:
        print('Restore train from cache!')
        # dictionary
        (train_data, train_target, driver_id, unique_drivers) = \
            restore_data(cache_path_w_file)
        print("type train_data, train_target: "+ str(type(train_data))+" "+str(type(train_target)))

    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    print("type train_data, train_target: "+ str(type(train_data))+" "+str(type(train_target)))

    if color_type == 1:
        train_data = train_data.reshape(train_data.shape[0], color_type,
                                        img_rows, img_cols)
    else:
        train_data = train_data.transpose((0, 3, 1, 2))

    train_target = np_utils.to_categorical(train_target, 10)
    train_data = train_data.astype('float32')
    print("type train_data, train_target: "+ str(type(train_data))+" "+str(type(train_target)))
    mean_pixel = [103.939, 116.779, 123.68]
    for c in xrange(3):
        train_data[:, c, :, :] = train_data[:, c, :, :] - mean_pixel[c]
    # train_data /= 255
    perm = permutation(len(train_target))
    train_data = train_data[perm]
    train_target = train_target[perm]
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    print("...exit read_and_normalize_and_shuffle_train_data...")
    return train_data, train_target, driver_id, unique_drivers
###}}}


def read_and_normalize_test_data(cache_name, img_rows=224, img_cols=224, color_type=1):
###{{{
    print("...read and normalize test data...")
    cache_path = "/home/ml0501/gabi/VGG16/cache18/test_r_224_c_224_t_3.datfile666.dat"
    # cache_path = os.path.join(str(cache_name), 'test_r_' + str(img_rows) +
                              # '_c_' + str(img_cols) + '_t_' +
                              # str(color_type) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        test_data, test_id = load_test(img_rows, img_cols, color_type)
        cache_data((test_data, test_id), cache_path, 666)
    else:
        print('Restore test from cache!')
        (test_data, test_id) = restore_data(cache_path)

    test_data = np.array(test_data, dtype=np.uint8)

    if color_type == 1:
        test_data = test_data.reshape(test_data.shape[0], color_type,
                                      img_rows, img_cols)
    else:
        test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float32')
    mean_pixel = [103.939, 116.779, 123.68]
    for c in xrange(3):
        test_data[:, c, :, :] = test_data[:, c, :, :] - mean_pixel[c]
    # test_data /= 255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print("...exit read_and_normalize_test_data...")
    return test_data, test_id
###}}}


def merge_several_folds_mean(data, nfolds):
###{{{
    a = np.array(data[0])

    for i in xrange(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    print("...exit merge_several_folds_mean...")
    return a.tolist()
###}}}


def vgg_std16_model(img_rows, img_cols, color_type=1):
###{{{
    print("...start vgg_std16_model...")
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(color_type,
                                                 img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    gabi4 = "/home/ml0501/Downloads/vgg16_weights.h5"
#model.load_weights('../input/vgg16_weights.h5')
    model.load_weights(gabi4)

# Code above loads pre-trained data and
    model.layers.pop()
    model.add(Dense(10, activation='softmax'))
# Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    print("...exit vgg_std16_model...")
    return model
###}}}


def vgg_std16_model2(img_rows, img_cols, color_type=1):
###{{{
    print("...start vgg_std16_model...")
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(color_type,
                                                 img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    gabi5 = "/home/ml0501/gabi/VGG16/cache18/model_weights2016-06-18.h5" #piemel
#model.load_weights('../input/vgg16_weights.h5')
    model.load_weights(gabi5)

# Code above loads pre-trained data and
    model.layers.pop()
    model.add(Dense(10, activation='softmax'))
# Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    print("...exit vgg_std16_model...")
    return model
###}}}


def my_generator(X_train, y_train, batch_size):
###{{{
    size_X_train = len(X_train)
    meta_batch = size_X_train / batch_size
    print("size_X_train: "+str(size_X_train))
    while 1:
        for i in range(meta_batch): # 1875 * 32 = 60000 -> # of training samples
            if i%100==0:
                print "i = " + str(i)
            yield X_train[i*batch_size:(i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size]
    pass
###}}}


# def shuffle_data(train_data, train_target,img_rows, img_cols, color_type=1):
def shuffle_data(train_data, train_target):
        # shuf_train_data_list, shuf_train_target_list = shuffle_data(train_data_list, train_target_list)
###{{{ 
    train_data = np.array(train_data)
    train_target = np.array(train_target)
    # print("train data shape")
    # print(train_data.shape)

    # if color_type == 1:
        # print("train_data.shape[0]: ")
        # print(train_data.shape[0])
        # train_data = train_data.reshape(train_data.shape[0], color_type,
                                        # img_rows, img_cols)
    # else:
        # train_data = train_data.transpose((0, 3, 1, 2))

    # train_target = np_utils.to_categorical(train_target, 10)
    # # train_data = train_data.astype('float32')
    # mean_pixel = [103.939, 116.779, 123.68]
    # print("type target, data: ")
    # print(type(train_target))
    # print(type(train_data))
    # print("shape target, data: ")
    # print(train_target.shape)
    # print(train_data.shape)
    # for c in xrange(3):
        # train_data[:, c, :, :] = train_data[:, c, :, :] - mean_pixel[c]
    # # train_data[:, 0, :, :] = train_data[:, 0, :, :] - mean_pixel[0]
    perm = permutation(len(train_target))
    train_data = train_data[perm]
    train_target = train_target[perm]
    return train_data, train_target
    pass
###}}}


def only_load_data(b, e):
###{{{
    print("...only loading data...")

    img_rows, img_cols = 224, 224
    # color_type = 1
    # if folder contains this, then it should have the others as well
    cache_path_w_file = '/home/ml0502/cache/file0.dat'
    cache_path_only_folder = '/home/ml0502/cache/'
    #n: number of pkl files we'l have
    n = 20
    #nn: number of pkl files we wanna load
    # nn = 10
    empty = ()

    # train_data, train_target, driver_id, unique_drivers = read_and_normalize_and_shuffle_train_data(img_rows, img_cols, color_type_global)
    print("...check if data is cached...")
    print(str(os.path.isfile(cache_path_w_file)))
    if not os.path.isfile(cache_path_w_file):
        print("...data not in cache, loading list...")
        # load training data into memory
        num_folders = 10
        p = "/home/ml0505/train/"
        train_data_list, train_target_list, driver_id, unique_drivers = load_train_as_list(p, num_folders)
        #shuffle
        print("...shuffle list...")
        shuf_train_data_list, shuf_train_target_list = shuffle_data(train_data_list, train_target_list)
        #store as pkl
        print("...store list in pkl batches...")
        num_imgs = len(shuf_train_data_list)
        print("num_imgs = "+str(num_imgs))
        for i in xrange(n):
            batch_data = shuf_train_data_list[i*(num_imgs/n):(i+1)*(num_imgs/n)]
            batch_target = shuf_train_target_list[i*(num_imgs/n):(i+1)*(num_imgs/n)]
            cache_data_from_list((batch_data, batch_target), cache_path_only_folder, i, color_type_global)
        #endfor
#{{{
        # train_data, train_target, driver_id, unique_drivers = \
            # load_train(img_rows, img_cols, color_type_global)
        # print("...shuffling data...")
        # shuf_train_data, shuf_train_target = \
                # shuffle_data(train_data, train_target,img_rows, img_cols, color_type_global)
        # print("type shuf target, data: ")
        # print(type(shuf_train_target))
        # print(type(shuf_train_data))
        # print("shape target, data: ")
        # print(shuf_train_target.shape)
        # print(shuf_train_data.shape)
        # print("...caching shuffled data...")
        # num_imgs = shuf_train_data.shape[0]
        # print("imgs per pkl: "+str(num_imgs/n))
        # for i in xrange(n):
            # batch_data = shuf_train_data[i*(num_imgs/n):(i+1)*(num_imgs/n)]
            # batch_target = shuf_train_target[i*(num_imgs/n):(i+1)*(num_imgs/n)]
            # cache_data((batch_data, batch_target), cache_path_only_folder, i)
        # #endfor
        # return shuf_train_data, shuf_train_target
###}}}

    #load from pkl
    # else:
    print('...restore train from cache!...')
    data = ()

    # tmp = ()
    print("...loading "+str(e-b)+" pkl files")
    for i in xrange(b,e):
        path = '/home/ml0502/cache/file' +str(i)+'.dat'
        tmp = restore_data2(path)
        print("type tmp:"+str(type(tmp))+". tmp[0]:" +str(tmp[0].shape)+". tmp[1]: "+str(tmp[1].shape))
        # print(np.shape(tmp))
        if data != empty:
            print("appending to data...")
            print("BEFORE data appended.")
            print("shape data: "+str(data[0].shape)+", "+str(data[1].shape))
            data_tmp = np.append(data[0], tmp[0], axis=0)
            target_tmp = np.append(data[1], tmp[1], axis=0)
            data = (data_tmp, target_tmp)
            print("AFTER data appended.")
            print("shape data: "+str(data[0].shape)+", "+str(data[1].shape))
            # print("data:")
            # print(data)
            # data[1].append(tmp[1])
        else:
            print("for some reason you're here")
            data = tmp
    #endfor
    print("data cached!")
    print(type(data))
    train_data, train_target = data[0], data[1]
    print(type(train_data))
    print(train_data.shape)
    print(type(train_target))
    print(train_target.shape)

    print("...exit only_load_data...")
    return train_data, train_target
    pass
###}}}


def train_model(nb_epoch):
###{{{
    img_rows, img_cols = 224, 224
    batch_size = 32
    random_state = 20
    print("...enter train model...")
    #img_rows, img_cols = 224, 224

    # train_data, train_target, driver_id, unique_drivers = \
        # read_and_normalize_and_shuffle_train_data(img_rows, img_cols,
                                                  # color_type_global)
    model = vgg_std16_model(img_rows, img_cols, color_type_global)
    print("train settings: train_data, train_target = only_load_data(0,5)")
    train_data, train_target = only_load_data(0,3)
    print("type train_data, train_target: "+ str(type(train_data))+" "+str(type(train_target)))
    model.fit_generator(my_generator(train_data, train_target, batch_size), samples_per_epoch = len(train_data), nb_epoch=nb_epoch, verbose=2, show_accuracy=True, callbacks=[], validation_data=None, class_weight=None, nb_worker=1)
    del train_data, train_target

    train_data, train_target = only_load_data(3,6)
    print("type train_data, train_target: "+ str(type(train_data))+" "+str(type(train_target)))
    model.fit_generator(my_generator(train_data, train_target, batch_size), samples_per_epoch = len(train_data), nb_epoch=nb_epoch, verbose=2, show_accuracy=True, callbacks=[], validation_data=None, class_weight=None, nb_worker=1)
    del train_data, train_target

    # train_data, train_target = only_load_data(6,9)
    # print("type train_data, train_target: "+ str(type(train_data))+" "+str(type(train_target)))
    # model.fit_generator(my_generator(train_data, train_target, batch_size), samples_per_epoch = len(train_data), nb_epoch=nb_epoch, verbose=2, show_accuracy=True, callbacks=[], validation_data=None, class_weight=None, nb_worker=1)
    # del train_data, train_target
    # train_data, train_target = only_load_data(8,9)
    # print("type train_data, train_target: "+ str(type(train_data))+" "+str(type(train_target)))
    # model.fit_generator(my_generator(train_data, train_target, batch_size), samples_per_epoch = len(train_data), nb_epoch=nb_epoch, verbose=2, show_accuracy=True, callbacks=[], validation_data=None, class_weight=None, nb_worker=1)
    # del train_data, train_target
    #model.train_on_batch(train_data, train_target)
    #model.fit(train_data, train_target, batch_size=bs,
                              # nb_epoch=nb_epoch,
                              # show_accuracy=True, verbose=1,
                              # validation_split=split, shuffle=True)



    try:
        print("try to save model")
        save_model2(model)
        print("save successful")
    except:
        print("save unsuccessful")

    start = 1
    end = 1
    #img_rows, img_cols = 224, 224
    # batch_size = 64
    # random_state = 51
    nb_epoch = 15

    print('Start testing............')
    test_data, test_id = read_and_normalize_test_data(img_rows, img_cols,
                                                      color_type_global)


    yfull_test = []

    for index in xrange(start, end + 1):
        # Store test predictions
        #model = read_model2()
        test_prediction = model.predict(test_data, batch_size=1, verbose=1)
        yfull_test.append(test_prediction)

    modelStr = ''
    info_string = 'loss_' + modelStr \
                  + '_r_' + str(img_rows) \
                  + '_c_' + str(img_cols) \
                  + '_folds_' + str(end - start + 1) \
                  + '_ep_' + str(nb_epoch)

    test_res = merge_several_folds_mean(yfull_test, end - start + 1)
    create_submission(test_res, test_id, info_string)
    print(info_string)
    print("...finished testing...")

    #print('Start testing............')
    #test_data, test_id = read_and_normalize_test_data(img_rows, img_cols,
    #                                                  color_type_global)
    #test_prediction = model.predict(test_data, batch_size=bs, verbose=1)
    #create_submission2(test_prediction, test_id)
    print("...exit train model...")
    pass
    ###}}}


def test_model_and_submit(start=1, end=1, modelStr=''):
###{{{
    img_rows, img_cols = 224, 224
# batch_size = 64
# random_state = 51
    nb_epoch = 15

    print('Start testing............')
    test_data, test_id = read_and_normalize_test_data(img_rows, img_cols,
                                                      color_type_global)
    yfull_test = []

    for index in xrange(start, end + 1):
        # Store test predictions
        model = read_model(index, modelStr)
        test_prediction = model.predict(test_data, batch_size=128, verbose=1)
        yfull_test.append(test_prediction)

    info_string = 'loss_' + modelStr \
                  + '_r_' + str(img_rows) \
                  + '_c_' + str(img_cols) \
                  + '_folds_' + str(end - start + 1) \
                  + '_ep_' + str(nb_epoch)

    test_res = merge_several_folds_mean(yfull_test, end - start + 1)
    create_submission(test_res, test_id, info_string)
    print(info_string)
    print("...exit test_model_and_submit...")
    pass
###}}}


def test_model_and_submit2():
###{{{
    start = 1
    end = 1
    img_rows, img_cols = 224, 224
    # batch_size = 64
    # random_state = 51
    nb_epoch = 15
    cache_name = "cache18"

    print('Start testing............')
    test_data, test_id = read_and_normalize_test_data(cache_name, img_rows, img_cols,
                                                      color_type_global)
    yfull_test = []
    print("...got test data...")

    for index in xrange(start, end + 1):
        # Store test predictions
        # model = read_model2()
        print("making network")
        model = vgg_std16_model2(img_rows, img_cols, color_type=1)
        test_prediction = model.predict(test_data, batch_size=1, verbose=1)
        yfull_test.append(test_prediction)

    modelStr = ''
    info_string = 'loss_' + modelStr \
                  + '_r_' + str(img_rows) \
                  + '_c_' + str(img_cols) \
                  + '_folds_' + str(end - start + 1) \
                  + '_ep_' + str(nb_epoch)

    test_res = merge_several_folds_mean(yfull_test, end - start + 1)
    create_submission(test_res, test_id, info_string)
    print(info_string)
    print("...exit test_model_and_submit...")
    ###}}}

#nb_epochs
# only_load_data()
train_model(3)
# path = '/home/ml0502/cache/'
# data=[]
# cache_data(data, path)
# test_model_and_submit2()
#test_model_and_submit(2, 3, 'high_epoch')

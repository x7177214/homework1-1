# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/ResNeXt-in-tensorflow
# ==============================================================================
import tarfile
from six.moves import urllib
import sys
import numpy as np
import cPickle
import os
from os import listdir
from os.path import isfile, join
import random
import skimage.io as io
import skimage.transform 
import tensorflow as tf

SCALE_FACTOR = 10 # downsampling scaling factor for training and valid
TEST_SCALE_FACTOR = 6 # downsampling scaling factor for testing

IMG_RAW_WIDTH = 1920
IMG_RAW_HEIGHT = 1080

IMG_TMP_WIDTH = IMG_RAW_WIDTH / 2 # temporal size saved in tfrecord
IMG_TMP_HEIGHT = IMG_RAW_HEIGHT / 2

IMG_TEST_WIDTH = IMG_RAW_WIDTH / TEST_SCALE_FACTOR
IMG_TEST_HEIGHT = IMG_RAW_HEIGHT / TEST_SCALE_FACTOR

IMG_WIDTH = IMG_RAW_WIDTH / SCALE_FACTOR
IMG_HEIGHT = IMG_RAW_HEIGHT / SCALE_FACTOR

IMG_DEPTH = 3

NUM_FA_CLASS = 2
NUM_GES_CLASS = 13
NUM_OBJ_CLASS = 24

TRAIN_EPOCH_SIZE = 14992
TEST_EPOCH_SIZE = 12776

def whitening_image(image_np, mode='test'):
    '''
    Performs per_image_whitening
    :param image_np: a 4D numpy array representing a batch of images
    :return: the image numpy array after whitened
    '''
    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        # Use adjusted standard deviation here, in case the std == 0.
        if mode is 'test':
            std = np.max([np.std(image_np[i, ...]), 1.0/np.sqrt(IMG_TEST_HEIGHT * IMG_TEST_WIDTH * IMG_DEPTH)])
        else:
            std = np.max([np.std(image_np[i, ...]), 1.0/np.sqrt(IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH)])
        image_np[i,...] = (image_np[i, ...] - mean) / std
    return image_np

def read_path_and_label(train_or_test_folder):
    '''
    input: 'train' or 'test'. Specify which folder want to read
    output: (string, string, float, float, float)
            [(hand_path, head_path, FA_label, ges_label, obj_label),
             (hand_path, head_path, FA_label, ges_label, obj_label),
             ...
             (hand_path, head_path, FA_label, ges_label, obj_label)]
    '''
    def find_num_files(location, cur_folder_idx):
        '''
        location: 'house', 'lab', 'office'
        cur_folder_idx: current folder index
        train_or_test_folder: choose train or test folder
        '''

        current_path = root_path + '/' + location + '/' + cur_folder_idx + '/Lhand/'
        num_files = len([f for f in listdir(current_path) if isfile(join(current_path, f))])

        return num_files

    def read_labels(location, cur_folder_idx, left_or_right, offset):
        '''
        location: 'house', 'lab', 'office'
        cur_folder_idx: current folder index
        left_or_right: left or right hand
        offset: the offset of cur_folder_idx
        '''
        
        root_path = '/Disk2/cedl/handcam/labels' # @ AI
        # root_path = '../dataset/labels' # @ my PC

        current_path = root_path + '/' + location + '/'
        post_fix = left_or_right + str(offset + cur_folder_idx) + '.npy'

        label_fa = np.load(current_path + 'FA_' + post_fix)
        label_ges = np.load(current_path + 'ges_' + post_fix)
        label_obj = np.load(current_path + 'obj_' + post_fix)

        return label_fa, label_ges, label_obj


    location_list = ['house', 'lab', 'office']
    num_folders_per_location = [3, 4, 3]
    hand_list = [('Lhand', 'left'), ('Rhand', 'right')]

    imgs_hand_path_list = []
    imgs_head_path_list = []
    labels_fa = []
    labels_ges = []
    labels_obj = []
    

    root_path = '/Disk2/cedl/handcam/frames/' + train_or_test_folder # @ AI
    # root_path = '../dataset/frames/' + train_or_test_folder # @ my PC

    for location, num_folders in zip(location_list, num_folders_per_location):
        for i in xrange(num_folders):
            num_files = find_num_files(location, str(i+1))
            for which_hand, L_or_R in hand_list:
                for j in xrange(num_files):
                    # hand
                    current_path = root_path + '/' + location + '/' + str(i+1) + '/' + which_hand + '/' 
                    imgs_hand_path_list.extend([current_path + 'Image' + str(j+1) + '.png'])
                    # head
                    current_path = root_path + '/' + location + '/' + str(i+1) + '/head/'
                    imgs_head_path_list.extend([current_path + 'Image' + str(j+1) + '.png'])
                # Labels
                # offset: label file idx. 
                # 0 for training data; num_folders_per_location for testing data
                if train_or_test_folder is 'train':
                    offset = 0
                elif train_or_test_folder is 'test':
                    offset = num_folders
                label_fa, label_ges, label_obj = read_labels(location, i+1, L_or_R, offset) 
                labels_fa.extend(label_fa)
                labels_ges.extend(label_ges)
                labels_obj.extend(label_obj)

    example = zip(imgs_hand_path_list, imgs_head_path_list, labels_fa, labels_ges, labels_obj)
    example = random.sample(example, len(example)) # shuffle the list

    return example

def read_in_imgs(imgs_path_list, mode):
    """
    This function reads all training or validation data, and returns the
    images as numpy arrays
    :param address_list: a list of paths of image files
    :return: concatenated numpy array of data. Data are in 4D arrays: [num_images,
    image_height, image_width, image_depth]
    """

    if mode is 'test':
        height = IMG_TEST_HEIGHT
        width = IMG_TEST_WIDTH
    else: # for valid or train
        height = IMG_HEIGHT
        width = IMG_WIDTH

    images = np.array([]).reshape([0, height, width, IMG_DEPTH])

    for imgs_path in imgs_path_list:
        img = io.imread(imgs_path) 
        img = skimage.transform.resize(img, [height, width], order=3, mode='reflect')
        if mode is 'train':
            img = horizontal_flip(image=img, axis=1) # 50% chance to flip the image when training
        img = np.reshape(img, [1, height, width, IMG_DEPTH])
        # Concatenate along axis 0 by default
        images = np.concatenate((images, img))

    return images


def tfrecords_maker(example):

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    tfrecords_filename = 'training_data.tfrecords'

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    i = 0
    for img_hand_path, img_head_path, label_fa, label_ges, label_obj in example:
        
        img_hand = np.array(io.imread(img_hand_path)) 
        img_head = np.array(io.imread(img_head_path))
        
        # half the image size to save storage
        img_hand = skimage.transform.resize(img_hand, [IMG_TMP_HEIGHT, IMG_TMP_WIDTH], order=3, mode='reflect')
        img_head = skimage.transform.resize(img_head, [IMG_TMP_HEIGHT, IMG_TMP_WIDTH], order=3, mode='reflect')

        img_hand = img_hand * 255.0
        img_head = img_head * 255.0
        img_hand = img_hand.astype(np.uint8)
        img_head = img_head.astype(np.uint8)

        image_hand_raw = img_hand.tostring()
        image_head_raw = img_head.tostring()
        
        _example = tf.train.Example(features=tf.train.Features(feature={
            'image_hand_raw': _bytes_feature(image_hand_raw),
            'image_head_raw': _bytes_feature(image_head_raw),
            'label_fa': _int64_feature(int(label_fa)),
            'label_ges': _int64_feature(int(label_ges)),
            'label_obj': _int64_feature(int(label_obj))}))
        
        writer.write(_example.SerializeToString())
        i = i + 1
        if i % 50 ==0:
            print '%d / %d' % (i, TRAIN_EPOCH_SIZE)
    writer.close()

if __name__ == '__main__':
    # To save the training data to tfrecord format
    train_data_list = read_path_and_label('train')
    tfrecords_maker(train_data_list)
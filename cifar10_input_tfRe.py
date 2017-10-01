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
# import cv2

data_dir = 'cifar10_data'
full_data_dir = 'cifar10_data/cifar-10-batches-py/data_batch_'
vali_dir = 'cifar10_data/cifar-10-batches-py/test_batch'
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

SCALE_FACTOR = 10 # downsampling scale

IMG_RAW_WIDTH = 1920
IMG_RAW_HEIGHT = 1080

IMG_TMP_WIDTH = IMG_RAW_WIDTH / 2 # temporal size saved in tfrecord
IMG_TMP_HEIGHT = IMG_RAW_HEIGHT / 2

IMG_WIDTH = IMG_RAW_WIDTH / SCALE_FACTOR
IMG_HEIGHT = IMG_RAW_HEIGHT / SCALE_FACTOR
IMG_DEPTH = 3

NUM_FA_CLASS = 2
NUM_GES_CLASS = 13
NUM_OBJ_CLASS = 24


# NUM_TRAIN_BATCH = 5 # How many batches of files you want to read in, from 0 to 5)
# EPOCH_SIZE = 10000 * NUM_TRAIN_BATCH # 10000: 10000 images per TRAIN_BATCH
TRAIN_EPOCH_SIZE = 14992
TEST_EPOCH_SIZE = 12776


def maybe_download_and_extract():
    '''
    Will download and extract the cifar10 data automatically
    :return: nothing
    '''
    dest_directory = data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size)
                                                             / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def _read_one_batch(path, is_random_label):
    '''
    The training data contains five data batches in total. The validation data has only one
    batch. This function takes the directory of one batch of data and returns the images and
    corresponding labels as numpy arrays
    :param path: the directory of one batch of data
    :param is_random_label: do you want to use random labels?
    :return: image numpy arrays and label numpy arrays
    '''
    fo = open(path, 'rb')
    dicts = cPickle.load(fo)
    fo.close()

    data = dicts['data']
    if is_random_label is False:
        label = np.array(dicts['labels'])
    else:
        labels = np.random.randint(low=0, high=10, size=10000)
        label = np.array(labels)
    return data, label


def read_in_all_images(address_list, shuffle=True, is_random_label = False):
    """
    This function reads all training or validation data, shuffles them if needed, and returns the
    images and the corresponding labels as numpy arrays
    :param address_list: a list of paths of cPickle files
    :return: concatenated numpy array of data and labels. Data are in 4D arrays: [num_images,
    image_height, image_width, image_depth] and labels are in 1D arrays: [num_images]
    """
    data = np.array([]).reshape([0, IMG_WIDTH * IMG_HEIGHT * IMG_DEPTH])
    label = np.array([])

    for address in address_list:
        print 'Reading images from ' + address
        batch_data, batch_label = _read_one_batch(address, is_random_label)
        # Concatenate along axis 0 by default
        data = np.concatenate((data, batch_data))
        label = np.concatenate((label, batch_label))

    num_data = len(label)

    # This reshape order is really important. Don't change
    # Reshape is correct. Double checked
    data = data.reshape((num_data, IMG_HEIGHT * IMG_WIDTH, IMG_DEPTH), order='F')
    data = data.reshape((num_data, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))


    if shuffle is True:
        print 'Shuffling'
        order = np.random.permutation(num_data)
        data = data[order, ...]
        label = label[order]

    data = data.astype(np.float32)
    return data, label


def horizontal_flip(image, axis):
    '''
    Flip an image at 50% possibility
    :param image: a 3 dimensional numpy array representing an image
    :param axis: 0 for vertical flip and 1 for horizontal flip
    :return: 3D image after flip
    '''
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
        # image = cv2.flip(image, axis)
        image = np.flip(image, axis)

    return image


def whitening_image(image_np):
    '''
    Performs per_image_whitening
    :param image_np: a 4D numpy array representing a batch of images
    :return: the image numpy array after whitened
    '''
    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        # Use adjusted standard deviation here, in case the std == 0.
        std = np.max([np.std(image_np[i, ...]), 1.0/np.sqrt(IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH)])
        image_np[i,...] = (image_np[i, ...] - mean) / std
    return image_np


def random_crop_and_flip(batch_data, padding_size):
    '''
    Helper to random crop and random flip a batch of images
    :param padding_size: int. how many layers of 0 padding was added to each side
    :param batch_data: a 4D batch array
    :return: randomly cropped and flipped image
    '''
    cropped_batch = np.zeros(len(batch_data) * IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH).reshape(
        len(batch_data), IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset+IMG_HEIGHT,
                      y_offset:y_offset+IMG_WIDTH, :]

        cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)

    return cropped_batch


def prepare_train_data(padding_size):
    '''
    Read all the train data into numpy array and add padding_size of 0 paddings on each side of the
    image
    :param padding_size: int. how many layers of zero pads to add on each side?
    :return: all the train data and corresponding labels
    '''
    path_list = []
    for i in range(1, NUM_TRAIN_BATCH+1):
        path_list.append(full_data_dir + str(i))
    data, label = read_in_all_images(path_list, is_random_label=TRAIN_RANDOM_LABEL)

    pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
    data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)

    return data, label


def read_validation_data():
    '''
    Read in validation data. Whitening at the same time
    :return: Validation image data as 4D numpy array. Validation labels as 1D numpy array
    '''
    validation_array, validation_labels = read_in_all_images([vali_dir],
                                                       is_random_label=VALI_RANDOM_LABEL)
    validation_array = whitening_image(validation_array)

    return validation_array, validation_labels

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
        # root_path = '/Disk2/cedl/handcam/frames/' + train_or_test_folder # @ AI
        # # root_path = '../dataset/frames/' + train_or_test_folder
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

def read_in_imgs(imgs_path_list, train_or_valid):
    """
    This function reads all training or validation data, and returns the
    images as numpy arrays
    :param address_list: a list of paths of image files
    :return: concatenated numpy array of data. Data are in 4D arrays: [num_images,
    image_height, image_width, image_depth]
    """

    images = np.array([]).reshape([0, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

    for imgs_path in imgs_path_list:
        img = io.imread(imgs_path)
        img = skimage.transform.resize(img, [IMG_HEIGHT, IMG_WIDTH], order=3, mode='reflect')
        if train_or_valid is 'train':
            img = horizontal_flip(image=img, axis=1) # 50% chance to flip the image when training
        img = np.reshape(img, [1, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
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
import numpy as np
from PIL import Image
from random import shuffle

from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras import applications
from sklearn.preprocessing import LabelBinarizer
from collections import deque

img_size_width = 64
img_size_height = 64
resize_algorithm = Image.BILINEAR
gen_pool_thresh = 8

"""
generate sea lion images with given yolo format.
imges should be in the folder: JPEGImages
labels should be in the folder: labels
"""
def sealion_gen(path_file, resize):
    ## end symbol is included in the line

    img_path_list = open(path_file).read().splitlines()
    # img_path_list = [line[:-1] for line in open(path_file)]
    shuffle(img_path_list)
    label_path_list = [(img_path[:-4].replace("JPEGImages", "labels") + "txt") for img_path in img_path_list]

    pair_pool = deque([])
    for index, img_path in enumerate(img_path_list):
        label_path = label_path_list[index]
        origin = Image.open(img_path)
        origin_w, origin_h = origin.size
        for line in open(label_path):
            line_split = line.split()
            kind, center_x_ratio, center_y_ratio, w_ratio, h_ratio = \
                int(line_split[0]), float(line_split[1]), float(line_split[2]), float(line_split[3]), float(line_split[4])
            center_x = origin_w * center_x_ratio
            center_y = origin_h * center_y_ratio
            half_w = origin_w * w_ratio * 0.5
            half_h = origin_h * h_ratio * 0.5
            box = (int(round(center_x - half_w)), int(round(center_y - half_h)), int(round(center_x + half_w)), int(round(center_y + half_h)))
            img = origin.crop(box)
            if resize:
                img = img.resize((img_size_width, img_size_height), resize_algorithm)
            pair_pool.append((img, kind))

        # encoder = LabelBinarizer()

        if len(pair_pool) >= gen_pool_thresh:
            tmp_x = []
            tmp_y = []
            counter = 0
            while counter < gen_pool_thresh:
                input, target = pair_pool.popleft()
                tmp_x.append(np.transpose(input, (2, 0, 1)))
                tmp_y.append(one_hot(target))
                counter += 1
            yield (np.array(tmp_x), np.array(tmp_y))


def one_hot(x):
    one_hot_array = np.array([0., 0., 0., 0., 0.])
    one_hot_array[x] = 1.
    return one_hot_array

"""
This is only a template
"""
def sealion_cnn():
    inputs = Input((1, 64, 64))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv2)

    flatten = Flatten()(pool2)
    dense1 = Dense(512, activation='relu')(flatten)
    dense2 = Dense(5, activation='softmax')(dense1)

    model = Model(input=inputs, output=dense2)
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

    return model


def sealion_vgg16(classes=5):
    img_input = Input((3, img_size_height, img_size_width))
    # Block 1
    x = Conv2D(3*32, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_first')(img_input)
    x = Conv2D(3*32, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format='channels_first')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format='channels_first')(x)

    # Block 2
    x = Conv2D(3*64, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format='channels_first')(x)
    x = Conv2D(3*64, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format='channels_first')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format='channels_first')(x)

    # Block 3
    x = Conv2D(3*128, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format='channels_first')(x)
    x = Conv2D(3*128, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format='channels_first')(x)
    x = Conv2D(3*128, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format='channels_first')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format='channels_first')(x)

    # # Block 4
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    #
    # # Block 5
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(3*512, activation='relu', name='fc1')(x)
    x = Dense(3*512, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(img_input, x, name='vgg16')
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_vgg16(train_generator, steps_per_epoch, validation_data=None):
    model = applications.VGG16(include_top=True, weights='imagenet', classes=5)
    model.fit_generator(generator=train_generator, steps_per_epoch=1, epochs=400, validation_data=None)


def load_model(path, model):
    model.load_weights(path)
    return model

############################
##          Main          ##
############################
if __name__ == '__main__':
    train_path = "/Users/ibm/GitRepo/Sealion/data/train.txt"
    train_gen = sealion_gen(train_path, resize=True)
    # batch = train_gen.next()
    # batch2 = train_gen.next()
    print "gen fininsh"

    # print len(batch)  #code for testing generator

    # print label
    # img.save('./sample.JPEG', 'JPEG')
    # print img.shape

    model = sealion_vgg16(5)
    model.fit_generator(generator=train_gen, steps_per_epoch=8, epochs=2, validation_data=None)


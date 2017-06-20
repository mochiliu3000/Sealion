import numpy as np
from PIL import Image
from random import shuffle, randint
import re

from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras import applications
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
from collections import deque

img_size_width = 64
img_size_height = 64
resize_algorithm = Image.BILINEAR
gen_pool_thresh = 32

"""
generate sea lion images with given yolo format.
imges should be in the folder: JPEGImages
labels should be in the folder: labels
"""
def sealion_gen(path_file, resize, max_num_per_class=60000, max_num_per_pos_img=5, max_num_per_neg_img=10):
    ## end symbol is included in the line
    img_path_list = open(path_file).read().splitlines()
    # img_path_list = [line[:-1] for line in open(path_file)]
    shuffle(img_path_list)
    label_path_list = [(img_path[:-4].replace("JPEGImages", "labels") + "txt") for img_path in img_path_list] # Need to make sure the image is stored in JPEGImages, while labels are stored in labels

    pair_pool = deque([])
    counter_list = [0] * 5 # Global counter list, count how many 64*64 images of each class has been appended into pair_pool
    pattern = re.compile(r'ng_[\d]+\.JPEG')

    img_path_list = img_path_list * 5
    label_path_list = label_path_list * 5
    for index, img_path in enumerate(img_path_list):
        print(">>>>>>>>>>" + str(counter_list))
        # Parse the img_path, if it has 'ng_[0-9]+.JPEG', it is high weighted negative image
        if pattern.match(img_path) != None and counter_list[4] < max_num_per_class and False:
            # Randomly gen small image, num <= max_num_per_neg_img
            imgs = fetch_random_img(img_path, max_num_per_neg_img, img_size_width)
            for img in imgs:
                pair_pool.append((img, 4))
            counter_list[4] += max_num_per_neg_img
            continue

        # Check the num of lines, if zero, it is normal weighted negative image
        num_lines = sum(1 for line in open(label_path_list[index]))
        if num_lines == 0 and counter_list[4] < max_num_per_class and False:
            # Randomly gen small image, num <= max_num_per_pos_img
            imgs = fetch_random_img(img_path, max_num_per_pos_img, img_size_width)
            for img in imgs:
                pair_pool.append((img, 4))
            counter_list[4] += max_num_per_pos_img
            continue

        label_path = label_path_list[index]
        origin = Image.open(img_path)
        origin_w, origin_h = origin.size
        counter_list_this_img = [0] * 4 
        for line in open(label_path):
            line_split = line.split()
            kind, center_x_ratio, center_y_ratio, w_ratio, h_ratio = \
                int(line_split[0]), float(line_split[1]), float(line_split[2]), float(line_split[3]), float(line_split[4])
            if counter_list[kind] >= max_num_per_class or counter_list_this_img[kind] >= max_num_per_pos_img: # if already get too many image for this sealion class or for this img
                print("WARN: Too many cuts for img %s, and for class %s Skip 1 line ..." % (label_path, kind))
                continue
            center_x = origin_w * center_x_ratio
            center_y = origin_h * center_y_ratio
            half_w = origin_w * w_ratio * 0.5
            half_h = origin_h * h_ratio * 0.5
            box = (int(round(center_x - half_w)), int(round(center_y - half_h)), int(round(center_x + half_w)), int(round(center_y + half_h)))
            img = origin.crop(box)
            if resize:
                img = img.resize((img_size_width, img_size_height), resize_algorithm)
            pair_pool.append((img, kind))
            counter_list[kind] += 1
            counter_list_this_img[kind] += 1

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


def fetch_random_img(img_path, num_fetch, size_fetch):
    origin = Image.open(img_path)
    origin_w, origin_h = origin.size
    half_size = int(size_fetch / 2)
    w_range = (half_size + 1, origin_w - half_size - 1)
    h_range = (half_size + 1, origin_h - half_size - 1)
    imgs = [] 
   
    for i in range(num_fetch):
        w_center = randint(*w_range)
        h_center = randint(*h_range)
        box = (w_center - half_size, h_center - half_size, w_center + half_size, h_center + half_size)
        img = origin.crop(box)
        imgs.append(img)
    return imgs

"""
generate and save label for classification. the first use case is yolo classification on a small training set
"""
def generate_and_save(path_file, img_dir, label_dir, resize=False, epohes=400, max_num_per_class=1000):
    train_gen = sealion_gen(path_file, resize, max_num_per_class=1000)
    counter = 0
    for _ in range(0, epohes):
        input_array, target_array = train_gen.next()
        for index, img in enumerate(input_array):
            kind = target_array[index]
            img.save(img_dir + '/' + 'classify_' + str(counter) + '.JPEG', 'JPEG')
            with open(label_dir + "/" + 'classify_' + str(counter) + ".txt", "wb") as file:
                file.write(str(kind) + " " + str(0.5) + " " + str(0.5) + " " + str(0.98) + " " + str(0.98) + "\n")
            counter += 1


"""
This is only a template
"""
def sealion_cnn(classes=5):
    inputs = Input((3, img_size_height, img_size_width))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    # pool2 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv2)

    flatten = Flatten()(conv2)
    dense1 = Dense(512, activation='relu')(flatten)
    dense1 = Dropout(0.3)(dense1)
    dense2 = Dense(classes, activation='softmax')(dense1)

    model = Model(input=inputs, output=dense2)
    model.summary()
    adam_d = Adam(lr=0.001, decay=1e-5)
    model.compile(loss='categorical_crossentropy',optimizer=adam_d, metrics=['accuracy'])

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
    train_path = "/home/hao/Desktop/sealion_training_data/out/all.txt"
    model_checkpoint_dir = "/home/hao/Desktop/sealion_training_data/sealion_vgg_weights"
    train_gen = sealion_gen(train_path, resize=True)
    # batch = train_gen.next()
    # batch2 = train_gen.next()
    print "gen fininsh"

    # print len(batch)  #code for testing generator

    # print label
    # img.save('./sample.JPEG', 'JPEG')
    # print img.shape

    model = sealion_cnn(5)
    # model = sealion_vgg16(5)
    # https://keras.io/callbacks/
    weight_save_callback = ModelCheckpoint(model_checkpoint_dir+"/weights.{epoch:04d}-{loss:.3f}.hdf5", monitor='loss', verbose=0, save_best_only=False, mode='auto', period=50)
    model.fit_generator(generator=train_gen, steps_per_epoch=32, epochs=3000, validation_data=None, callbacks=[weight_save_callback])


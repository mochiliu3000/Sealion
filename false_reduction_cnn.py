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

img_size_width = 64
img_size_height = 64
resize_algorithm = Image.BILINEAR

"""
generate sea lion images with given yolo format.
imges should be in the folder: JPEGImages
labels should be in the folder: labels
"""
def sealion_gen(path_file, resize):
    ## end symbol is included in the line
    img_path_list = [line[:-1] for line in open(path_file)]
    shuffle(img_path_list)
    label_path_list = [(img_path[:-4].replace("JPEGImages", "labels") + "txt") for img_path in img_path_list]

    for index, img_path in enumerate(img_path_list):
        pairs = []
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
            pairs.append((img, kind))
        for pair in pairs:
            yield pair

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
    dense2 = Dense(4, activation='softmax')(dense1)

    model = Model(input=inputs, output=dense2)
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

    return model

def load_model(path, model):
    model.load_weights(path)
    return model

############################
##          Main          ##
############################
if __name__ == '__main__':
    train_path = "/home/sleepywyn/Dev/GitRepo/Sealion/data/validate.txt"
    train_gen = sealion_gen(train_path, resize=True)
    img, label = train_gen.next()
    print img.size
    print label
    img.save('./sample.JPEG', 'JPEG')

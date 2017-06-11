import numpy as np
from PIL import Image
from random import shuffle

from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

"""
generate sea lion images with given yolo format.
imges should be in the folder: JPEGImages
labels should be in the folder: labels
"""
def sealion_gen(path_file):
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
            pairs.append((img, kind))
        for pair in pairs:
            yield pair



############################
##          Main          ##
############################
if __name__ == '__main__':
    train_path = "/home/sleepywyn/Dev/GitRepo/Sealion/data/validate.txt"
    train_gen = sealion_gen(train_path)
    img, label = train_gen.next()
    print img.size
    print label
    img.save('./sample.JPEG', 'JPEG')

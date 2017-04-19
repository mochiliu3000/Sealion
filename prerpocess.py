import numpy as np
import pandas as pd
import os
import skimage.feature
from sklearn.preprocessing import LabelBinarizer
import cv2
import csv
from PIL import Image

def extract_coords(train_path, train_dot_path, out_dir):
    file_names = os.listdir(train_path)
    for filename in file_names:
        # filename = "41_3.jpeg"          # Testing a bug
        print("processing image: " + str(filename))
        # read the Train and Train Dotted images
        try:
            image_1 = cv2.imread(train_path + "/" + filename)
            image_2 = cv2.imread(train_dot_path + "/" + filename)
        except: continue
        # absolute difference between Train and Train Dotted
        image_3 = cv2.absdiff(image_1, image_2)

        # mask out blackened regions from Train Dotted
        mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        mask_1[mask_1 < 20] = 0
        mask_1[mask_1 > 0] = 255

        mask_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
        mask_2[mask_2 < 20] = 0
        mask_2[mask_2 > 0] = 255

        image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_1)
        image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_2)

        # convert to grayscale to be accepted by skimage.feature.blob_log
        image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)

        fn_splt = filename.split('.')
        basename = fn_splt[0]

        if np.count_nonzero(image_3) == 0:
            open(out_dir + "/" + basename + ".txt", "wb").close()
            continue   # skip
        # detect blobs
        blobs = skimage.feature.blob_log(image_3, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02)
        print image_1.shape
        blob_num = len(blobs)

        w = image_1.shape[0]
        h = image_2.shape[1]

        with open(out_dir + "/" + basename + ".txt", "wb") as file:
            for blob in blobs:
                # get the coordinates for each blob
                y, x, s = blob
                x1, x2, y1, y2 = x-16, x+16, y-16, y+16
                # print x1, x2, y1, y2
                if x1 < 0 or y1 < 0 or x2 > image_1.shape[1] or y2 > image_1.shape[0]:
                    continue
                x_center, y_center, w_ratio, h_ratio = convert_coord(x, y, w, h)    # convert to ratio required by darknet
                # get the color of the pixel from Train Dotted in the center of the blob
                file.write("0" + " " + str(x_center) + " " + str(y_center) + " " + str(w_ratio) + " " + str(h_ratio) + "\n")
                # file.write("0" + " " + str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + "\n")


def convert_coord(x, y, w_img, h_img):
    dw = 1. / w_img
    dh = 1. / h_img
    x_center = x * dw
    y_center = y * dh
    w_ratio = 32 * dw
    h_ratio = 32 * dh
    return x_center, y_center, w_ratio, h_ratio

def splitimage(src, rownum, colnum, dstpath):
    ext = "jpeg"
    img = Image.open(src)
    w, h = img.size
    if rownum <= h and colnum <= w:
        print('Original image info: %sx%s, %s, %s' % (w, h, img.format, img.mode))
        print('Start spliting image')
        s = os.path.split(src)
        if dstpath == '':
            dstpath = s[0]
        fn = s[1].split('.')
        basename = fn[0]
        num = 0
        rowheight = h // rownum
        colwidth = w // colnum
        for r in range(rownum):
            for c in range(colnum):
                box = (c * colwidth, r * rowheight, (c + 1) * colwidth, (r + 1) * rowheight)
                img.crop(box).save(os.path.join(dstpath, basename + '_' + str(num) + '.' + ext), ext)
                num = num + 1

        print('Splitting complete.  %s images' % num)
    else:
        print('Invalid cut parameter')

            # decision tree to pick the class of the blob by looking at the color in Train Dotted
        #     if r > 200 and g < 50 and b < 50:  # RED
        #         adult_males.append((int(x), int(y)))
        #     elif r > 200 and g > 200 and b < 50:  # MAGENTA
        #         subadult_males.append((int(x), int(y)))
        #     elif r < 100 and g < 100 and 150 < b < 200:  # GREEN
        #         pups.append((int(x), int(y)))
        #     elif r < 100 and 100 < g and b < 100:  # BLUE
        #         juveniles.append((int(x), int(y)))
        #     elif r < 150 and g < 50 and b < 100:  # BROWN
        #         adult_females.append((int(x), int(y)))
        #
        # coordinates_df["adult_males"][filename] = adult_males
        # coordinates_df["subadult_males"][filename] = subadult_males
        # coordinates_df["adult_females"][filename] = adult_females
        # coordinates_df["juveniles"][filename] = juveniles
        # coordinates_df["pups"][filename] = pups

if __name__ == '__main__':
    train_path = "./data/TrainSmall2/Train_split"
    train_dot_path = "./data/TrainSmall2/TrainDotted_split"
    outdir = "./data/TrainSmall2/labels"

    extract_coords(train_path, train_dot_path, out_dir=outdir)
    # splitimage("./data/TrainSmall2/Train/41.jpg", 6, 6, "./data/TrainSmall2/Train_split")

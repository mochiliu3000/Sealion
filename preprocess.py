import numpy as np
import pandas as pd
import os
import skimage.feature
# from sklearn.preprocessing import LabelBinarizer
import cv2
import csv
import re
from PIL import Image
from sets import Set
from multiprocessing import Pool
from functools import partial


thread_num = 6
COUNTER = 0


def extract_coord(train_path, train_dot_path, out_dir, filename):
    # filename = "41_27.jpeg"          # Testing a bug
    print("processing image: " + str(filename))
    # read the Train and Train Dotted images
    try:
        image_1 = cv2.imread(train_dot_path + "/" + filename)
        image_2 = cv2.imread(train_path + "/" + filename)
    except:
        return
    # absolute difference between Train and Train Dotted
    image_3 = cv2.absdiff(image_1, image_2)
    # plt.imshow(image_1)
    # plt.show()
    # plt.imshow(image_2)
    # plt.show()
    # plt.imshow(image_3)
    # plt.show()

    # mask out blackened regions from Train Dotted
    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 50] = 0
    mask_1[mask_1 > 0] = 255
    # plt.imshow(mask_1)
    # plt.show()

    mask_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    mask_2[mask_2 < 50] = 0
    mask_2[mask_2 > 0] = 255
    # plt.imshow(mask_2)
    # plt.show()

    image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_1)
    image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_2)

    # plt.imshow(image_3)
    # plt.show()

    # convert to grayscale to be accepted by skimage.feature.blob_log
    image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)

    fn_splt = filename.split('.')
    basename = fn_splt[0]

    if np.count_nonzero(image_3) == 0:
        # open(out_dir + "/" + basename + ".txt", "wb").close()
        return  # skip
    # detect blobs
    blobs = skimage.feature.blob_log(image_3, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02)
    print image_1.shape
    blob_num = len(blobs)

    w = image_1.shape[1]
    h = image_1.shape[0]

    if blob_num > 0:
        with open(out_dir + "/" + basename + ".txt", "wb") as file:
            for blob in blobs:
                # get the coordinates for each blob
                y, x, s = blob
                # get the color of the pixel from Train Dotted in the center of the blob
                b, g, r = image_1[int(y)][int(x)][:]
                # decision tree to pick the class of the blob by looking at the color in Train Dotted
                if r > 200 and b < 50 and g < 50: # RED adult_males 67.4
                    x1, x2, y1, y2 = x - 67.4/2.0, x + 67.4/2.0, y - 67.4/2.0, y + 67.4/2.0
                    inx = 0
                elif r > 200 and b > 200 and g < 50: # MAGENTA subadult_males 53.6
                    x1, x2, y1, y2 = x - 53.6/2.0, x + 53.6/2.0, y - 53.6/2.0, y + 53.6/2.0
                    inx = 1					
                elif r < 100 and b < 100 and 150 < g < 200: # GREEN pups 
                    print "Here is a pup, won't track it"
                    continue
                elif r < 100 and  100 < b and g < 100: # BLUE juveniles 34.0
                    x1, x2, y1, y2 = x - 34.0/2.0, x + 34.0/2.0, y - 34.0/2.0, y + 34.0/2.0
                    inx = 2
                elif r < 150 and b < 50 and g < 100:  # BROWN adult_females 53.6
                    x1, x2, y1, y2 = x - 53.6/2.0, x + 53.6/2.0, y - 53.6/2.0, y + 53.6/2.0
                    inx = 3					
                else:
                    print "Error point, the color is unknown... skip it for now"
                    continue
                             
                # print x1, x2, y1, y2
                if x1 < 0 or y1 < 0 or x2 > image_1.shape[1] or y2 > image_1.shape[0]:
                    print "Error point, the bbox is on the edge of image... skip it for now"
                    continue
                bbox_sizes = [67.4, 53.6, 34.0, 53.6]
                x_center, y_center, w_ratio, h_ratio = convert_coord(x, y, w, h, bbox_sizes[inx])  # convert to ratio required by darknet
                # get the color of the pixel from Train Dotted in the center of the blob
                file.write(str(inx) + " " + str(x_center) + " " + str(y_center) + " " + str(w_ratio) + " " + str(h_ratio) + "\n")
                # file.write("0" + " " + str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + "\n")

        print("The blob num is "  + str(blob_num))
        with open(out_dir + "/all.txt", "a") as tr_file:
            tr_file.write(train_path + "/" + filename + "\n")


def extract_coords(train_path, train_dot_path, out_dir):
    file_names = os.listdir(train_path)
    func = partial(extract_coord, train_path, train_dot_path, out_dir)
    pool = Pool(thread_num)
    pool.map(func, file_names)
        # for filename in file_names:
        #     # filename = "41_27.jpeg"          # Testing a bug
        #     print("processing image: " + str(filename))
        #     # read the Train and Train Dotted images
        #     try:
        #         image_1 = cv2.imread(train_dot_path + "/" + filename)
        #         image_2 = cv2.imread(train_path + "/" + filename)
        #     except: continue
        #     # absolute difference between Train and Train Dotted
        #     image_3 = cv2.absdiff(image_1, image_2)
        #     # plt.imshow(image_1)
        #     # plt.show()
        #     # plt.imshow(image_2)
        #     # plt.show()
        #     # plt.imshow(image_3)
        #     # plt.show()
        #
        #     # mask out blackened regions from Train Dotted
        #     mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        #     mask_1[mask_1 < 50] = 0
        #     mask_1[mask_1 > 0] = 255
        #     # plt.imshow(mask_1)
        #     # plt.show()
        #
        #     mask_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
        #     mask_2[mask_2 < 50] = 0
        #     mask_2[mask_2 > 0] = 255
        #     # plt.imshow(mask_2)
        #     # plt.show()
        #
        #     image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_1)
        #     image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_2)
        #
        #     # plt.imshow(image_3)
        #     # plt.show()
        #
        #     # convert to grayscale to be accepted by skimage.feature.blob_log
        #     image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)
        #
        #     fn_splt = filename.split('.')
        #     basename = fn_splt[0]
        #
        #     if np.count_nonzero(image_3) == 0:
        #         open(out_dir + "/" + basename + ".txt", "wb").close()
        #         continue   # skip
        #     # detect blobs
        #     blobs = skimage.feature.blob_log(image_3, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02)
        #     print image_1.shape
        #     blob_num = len(blobs)
        #
        #     w = image_1.shape[1]
        #     h = image_1.shape[0]
        #
        #     with open(out_dir + "/" + basename + ".txt", "wb") as file:
        #         for blob in blobs:
        #             # get the coordinates for each blob
        #             y, x, s = blob
        #             x1, x2, y1, y2 = x-16, x+16, y-16, y+16
        #             # print x1, x2, y1, y2
        #             if x1 < 0 or y1 < 0 or x2 > image_1.shape[1] or y2 > image_1.shape[0]:
        #                 continue
        #             x_center, y_center, w_ratio, h_ratio = convert_coord(x, y, w, h)    # convert to ratio required by darknet
        #             # get the color of the pixel from Train Dotted in the center of the blob
        #             file.write("0" + " " + str(x_center) + " " + str(y_center) + " " + str(w_ratio) + " " + str(h_ratio) + "\n")
        #             # file.write("0" + " " + str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + "\n")
        #     if blob_num > 0:
        #         tr_file.write(train_path + "/" + filename + "\n")

def convert_coord(x, y, w_img, h_img, bbox_size):
    dw = 1. / w_img
    dh = 1. / h_img
    x_center = x * dw
    y_center = y * dh
    w_ratio = bbox_size * dw
    h_ratio = bbox_size * dh
    return x_center, y_center, w_ratio, h_ratio

def splitimage(rownum, colnum, dstpath, src):
    ext = "JPEG"
    global COUNTER
    img = Image.open(src)
    w, h = img.size
    if rownum <= h and colnum <= w:
        print('Original image info: %sx%s, %s, %s' % (w, h, img.format, img.mode))
        print('Start spliting image.')
        # COUNTER += 1
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

def split_images(src_folder, dst_folder, skip_set):
    all_img_set = Set(os.listdir(src_folder))
    images = all_img_set - skip_set

    images = map(lambda x: src_folder + "/" + x, images)

    print images

    func = partial(splitimage, 12, 16, dst_folder)
    pool = Pool(thread_num)
    pool.map(func, images)


def skip_img_set(path):
    skip_set = Set()
    with open(path) as f:
        next(f)
        [skip_set.add(img_id[:-1] + ".jpg") for img_id in f]
    return skip_set


def rotate_label(degree, x, y, width, height):
    if degree == 90:
        return y, 1-x, height, width
    if degree == 180:
        return 1-x, 1-y, width, height
    if degree == 270:
        return 1-y, x, height, width


def reverse_label(direction, x, y, width, height):
    if direction == "horizontal":
        return 1-x, y, width, height
    if direction == "vertical":
        return x, 1-y, width, height


def read_and_save(path_file, img_dir, label_dir):
    img_name_regex = re.compile("^.+/JPEGImages/(.+)\.jpg$")
    img_path_list = open(path_file).read().splitlines()
    label_path_list = [(img_path[:-3].replace("JPEGImages", "labels") + "txt") for img_path in img_path_list]

    for index, img_path in enumerate(img_path_list):
        img_name = img_name_regex.search(img_path).group(1)
        print "processing" + str(img_name)
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
        counter = 0
        for (img, kind) in pairs:
            img.save(img_dir + "/classify_" + img_name + "_" + str(counter) + ".JPEG", "JPEG")
            with open(label_dir + "/classify_" + img_name + "_" + str(counter) + ".txt", "wb") as file:
                file.write(str(kind) + " " + str(0.5) + " " + str(0.5) + " " + str(0.98) + " " + str(0.98) + "\n")
            counter += 1

if __name__ == '__main__':
    train_path = "./data/Train"
    train_dotted_path = "./data/TrainDotted"
    train_split_dst = "/home/sleepywyn/Dev/GitRepo/Sealion/data/JPEGImages"   # should be absolute path here
    train_dotted_split_dst = "/home/sleepywyn/Dev/GitRepo/Sealion/data/JPEGDottedImages"  # should be absolute path here
    label_dir = "/home/sleepywyn/Dev/GitRepo/Sealion/data/labels"
    skip_img_path = "./data/MismatchedTrainImages.txt"

    test_path = "./data/Test"
    test_split_dst = "/home/sleepywyn/Dev/GitRepo/Sealion/data/Test_split"

    # for splitting addon images
    # train_path = "./data/addon_samples/mismatch"
    # train_dotted_path = "./data/addon_samples/mismatchDotted"
    # train_split_dst = "/home/sleepywyn/Dev/GitRepo/Sealion/data/addon_samples/JPEGMismatch"   # should be absolute path here
    # train_dotted_split_dst = "/home/sleepywyn/Dev/GitRepo/Sealion/data/addon_samples/JPEGMismatchDotted"  # should be absolute path here
    # label_dir = "/home/sleepywyn/Dev/GitRepo/Sealion/data/addon_samples/labels"
    # skipped_img_ids = Set()

    # skipped_img_ids = skip_img_set(skip_img_path)
    # split_images(train_path, train_split_dst, skipped_img_ids)
    # split_images(train_dotted_path, train_dotted_split_dst, skipped_img_ids)

    # split_images(test_path, test_split_dst, skipped_img_ids)

    extract_coords(train_split_dst, train_dotted_split_dst, out_dir=label_dir)


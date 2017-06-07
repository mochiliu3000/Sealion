import os
import operator
import matplotlib.pylab as plt
import collections
import numpy as np

label_dir = "/home/sleepywyn/Dev/GitRepo/Sealion/data/labels"
image_dir = "/home/sleepywyn/Dev/GitRepo/Sealion/data/JPEGImages"
out_dir = "/home/sleepywyn/Dev/GitRepo/Sealion/data"


def gen_nonzero_label(label_dir, image_dir, out_dir):
    if not os.path.exists(label_dir):
        print("ERROR: The label folder you provided does not exists. Exit.")
    else:
        with open(out_dir + "/train.txt", "wb") as tr_file:
            for label_file in os.listdir(label_dir):
                num_lines = sum(1 for line in open(label_dir + "/" + label_file))
                if num_lines > 0:
                    tr_file.write(image_dir + "/" + label_file[:-3] + "JPEG\n")


def gen_less_female_nonzero_label(label_dir, image_dir, out_dir, threshold, hist=False):
    if not os.path.exists(label_dir) or not os.path.exists(image_dir):
        print("ERROR: Label or Image folder you provided does not exist. Exit.")
    else:
        f_dict = {}
        for label_file in os.listdir(label_dir):
            num_lines = sum(1 for line in open(label_dir + "/" + label_file))
            if num_lines > 0:
                labels = [line.split()[0] for line in open(label_dir + "/" + label_file)]
                num_f_lines = sum([int(i == '3') for i in labels])
                f_dict[label_file] = num_f_lines
        if hist:
            plt.hist(f_dict.values())
            plt.show()
        sorted_f_dict = sorted(f_dict.items(), key=operator.itemgetter(1))
        dict_leng = int(round(len(sorted_f_dict) * (1 - threshold)))
        remain_dict = sorted_f_dict[:dict_leng]

        print("Last image: ", remain_dict[-1])
        remain_imgs = [i[0] for i in remain_dict]
        with open(out_dir + "/train_new.txt", "wb") as tr_file:
            for label_file in remain_imgs:
                tr_file.write(image_dir + "/" + label_file[:-3] + "JPEG\n")


def gen_validation_label(label_dir, image_dir, out_dir, valid_perc = 0.02, no_sealion_perc = 0.1):
    if not os.path.exists(label_dir) or not os.path.exists(image_dir):
        print("ERROR: Label or Image folder you provided does not exist. Exit.")
    else:
        # Note: Each big picture has a type, for example, solid, stone, sea, wood ...
        pic_type = []   # length = 948
        type_dict = {}  # {solid: 0, stone: 1, grass: 2, sea: 3, animal&other: 4}
        pic_tc = collections.Counter(pic_type)
        print("INFO: Picture types and corresponding counts are ", pic_tc)
        print("INFO: The type names are ", type_dict)
        count_dict = {}  # store the number of each type sealions for each small picture (at least 1 sealion is in the pic)
        no_sealion_pic_list = []  # store the id of small pictures which have 0 sealion
        pic_count = 0  # store the number of small pictures

        for label_file in os.listdir(label_dir):
            num_lines = sum(1 for line in open(label_dir + "/" + label_file))
            if num_lines > 0:
                # collections.Counter generate dictionary like this: {0:2, 1:2, 2:0, 3:4}, sort it by key and fetch value
                labels_count = collections.Counter([line.split()[0] for line in open(label_dir + "/" + label_file)])
                sorted_count = sorted(labels_count.values(), key=operator.itemgetter(0))
                count_dict[label_file] = sorted_count  # adult_males, subadult_males, juveniles, adult_females
            else:
                no_sealion_pic_list.append(label_file)
            pic_count += 1
        print("INFO: Generated count_dict for each small pic, if there is no sealion in it, stored it into no_sealion_pic_list..." )
        print("INFO: Overall ", pic_count, " small pictures, ", len(no_sealion_pic_list), " of them has 0 sealions.")
        valid_size = round(pic_count * valid_perc)
        no_sealion_size = round(valid_size * no_sealion_perc)
        seaslion_size = valid_size - no_sealion_size

        # Note: Try to generate validation set with all kinds of pic_type and count_dict:
        # Hence for each kind of pic_type, each kind of sealion count threshold class, at 1 validation small picture will be selected
        # The sealion count threshold class here can be (small_number, large_number)
        # Hence in all we get: 2*2*2*2*5 possible combinations for pic with at least 1 sealion; and for no sealion pic, 5 possible pic_type
        pic_num_each_comb = seaslion_size / (2*2*2*2*5) + 1
        pic_num_each_pic_type = no_sealion_size / 5 + 1
        print("INFO: Searching small pictures with at least 1 sealion, for each combination,  ", pic_num_each_comb,
              " pictures will be randomly fetched...")
        print("INFO: Also Searching small pictures with 0 sealion, for each pic_type,  ", pic_num_each_pic_type,
              " pictures will be randomly fetched...")
        print("INFO: In all, plan to generate a validation set of size ", (pic_num_each_comb * (2*2*2*2*5) + pic_num_each_pic_type * 5),
              ", ", (pic_num_each_pic_type * 5), " of them has 0 sealions.")

        full_key_list = key_list = val_list = []
        for key, value in count_dict.items():
            full_key_list.append(key)  # store full key for later use
            key_list.append(pic_type[key.split("_")[0]])  # key is like "44_3" => 44 => a number in [0, 1, 2, 3, 4]
            val_list.append(value)  # value is like [1, 4, 3, 7]

        count_arr = np.array(value)
        mid_arr = np.median(count_arr, axis=0)  # calculate median value for each column (1 * 4)
        bool_arr = count_arr > mid_arr  # get a bool arr like [[T, F, F, T], [F, F, T, T], ...]
        val_arr = np.array(val_list)
        key_arr = np.array(full_key_list)
        result_set = set()

        for i in range(5):
            for j in range(16):
                bool_select = np.array([c == '1' for c in bin(j)[2:]])  # [T, F, F, T]
                typed_bool_arr = bool_arr[val_arr == i]
                ok_count = np.sum([(e == bool_select).all() for e in typed_bool_arr])
                if ok_count == 0:
                    print("WARN: No small picture match the requirement: pic_type=", i, " and bool_select=", bool_select)
                    continue
                elif ok_count < pic_num_each_comb:
                    print("WARN: Only find ", ok_count, " small pictures match the requirement (Need ", pic_num_each_comb,
                          " pictures): pic_type=", i, " and bool_select=", bool_select)
                else:
                    print("INFO: Find ", ok_count, " small pictures match the requirement: pic_type=", i,
                          " and bool_select=", bool_select)
                ok = [(e == bool_select).all() for e in typed_bool_arr]
                ok_labels = (key_arr[val_arr == i])[ok]
                result_set.add(ok_labels)

        print("INFO: Finally ...... generated a validation set of size ", len(result_set))
        return result_set



def batch_rename(path):
    for file_name in os.listdir(path):
        os.rename(path + "/" + file_name, path + "/" + file_name[:-3] + "JPEG")



if __name__ == '__main__':
    # gen_nonzero_label(label_dir, image_dir, out_dir)
    # batch_rename("./data/Yolo_mark_result/JPEGImages")
    gen_less_female_nonzero_label(label_dir, image_dir, out_dir, 0.3, True)

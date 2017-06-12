import os
from shutil import copyfile
import operator
import matplotlib.pylab as plt
import collections
import numpy as np

label_dir = "/home/hao/Desktop/sealion_training_data/labels"
image_dir = "/home/hao/Desktop/sealion_training_data/Train_split"
dot_image_dir = "/home/hao/Desktop/sealion_training_data/TrainDotted_split"
out_dir = "/home/hao/Desktop/sealion_training_data/out" # user defined to store validation images and labels

weight_dir = "/home/hao/Desktop/sealion_training_data/round4_weight"
darknet_dir = "/home/hao/darknet"
valid_txt_dir = out_dir + "/valid_sealion.txt"
neg_img_dir = "/home/hao/Desktop/sealion_training_data/negative_split" # put images under neg_img_dir/JPEGImages, labels under neg_img_dir/labels
valid_pred_dir = "/home/hao/Desktop/sealion_training_data/Valid_pred" # user defined to store validation predictions

def gen_nonzero_label(label_dir, image_dir, out_dir):
    if not os.path.exists(label_dir):
        print("ERROR: The label folder you provided does not exists. Exit.")
    else:
        with open(out_dir + "/train.txt", "wb") as tr_file:
            for label_file in os.listdir(label_dir):
                num_lines = sum(1 for line in open(label_dir + "/" + label_file))
                if num_lines > 0:
                    tr_file.write(image_dir + "/" + label_file[:-3] + "JPEG\n")

def gen_empty_label(image_dir, label_dir):
    img_list = os.listdir(image_dir)
    if not img_list:
        print("ERROR: The image folder you provided does not exists. Exit.")
    else:
        for img_name in img_list:
            label_path = label_dir + "/" + img_name[:-4] + "txt"
            with open(label_path, "wb") as tr_file:
                print "generating empty label for file: " + img_name


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


def gen_validation_label(label_dir, image_dir, dot_image_dir, out_dir, valid_perc = 0.003, no_sealion_perc = 0.1):
    if not os.path.exists(label_dir) or not os.path.exists(image_dir) or not os.path.exists(dot_image_dir):
        print("ERROR: Label or Image folder you provided does not exist. Exit.")
    else:
        # Note: Each big picture has a type, for example, solid, stone, sea, wood ...
        pic_type = ([0, 1, 2, 3, 4] * 190)[:-2]   # length = 948
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
                box_labels = np.array([line.split()[0] for line in open(label_dir + "/" + label_file)])
                sorted_count = [np.sum(box_labels == '0'), np.sum(box_labels == '1'),
                                np.sum(box_labels == '2'), np.sum(box_labels == '3')]
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
        pic_num_each_comb = int(seaslion_size / (2*2*2*2*5)) + 1
        pic_num_each_pic_type = int(no_sealion_size / 5) + 1
        print("INFO: Searching small pictures with at least 1 sealion, for each combination,  ", pic_num_each_comb,
              " pictures will be randomly fetched...")
        print("INFO: Also Searching small pictures with 0 sealion, for each pic_type,  ", pic_num_each_pic_type,
              " pictures will be randomly fetched...")
        print("INFO: In all, plan to generate a validation set of size ", (pic_num_each_comb * (2*2*2*2*5) + pic_num_each_pic_type * 5),
              ", ", (pic_num_each_pic_type * 5), " of them has 0 sealions.")

        full_key_list = []; key_list = []; val_list = []
        for key, value in count_dict.iteritems():
            if key == "all.txt":
                print("WARN: Find 'all.txt' in folder ", label_dir, " . Skip it...")
                continue
            full_key_list.append(key)  # store full key for later use
            key_list.append(pic_type[int(key.split("_")[0])])  # key is like "44_3.txt" => 44 => a number in [0, 1, 2, 3, 4]
            val_list.append(value)  # value is like [1, 4, 3, 7]

        count_arr = np.array(val_list)
        mid_arr = np.median(count_arr, axis=0)  # calculate median value for each column (1 * 4)
        bool_arr = count_arr > mid_arr  # get a bool arr like [[T, F, F, T], [F, F, T, T], ...]
        val_arr = np.array(key_list)
        key_arr = np.array(full_key_list)
        result_set = set()

        for i in range(5):
            for j in range(16):
                b_value = bin(j)[2:]
                b_value = '0' * (4 - len(b_value)) + b_value
                bool_select = np.array([c == '1' for c in b_value])  # [T, F, F, T]
                typed_bool_arr = bool_arr[val_arr == i]
                ok_ind = [(e == bool_select).all() for e in typed_bool_arr]
                ok_count = np.sum(ok_ind)
                if ok_count == 0:
                    print("WARN: No small picture match the requirement: pic_type=", i, " and bool_select=", bool_select)
                    continue
                elif ok_count < pic_num_each_comb:
                    print("WARN: Only find ", ok_count, " small pictures match the requirement (Need ", pic_num_each_comb,
                          " pictures): pic_type=", i, " and bool_select=", bool_select)
                    ok_labels = (key_arr[val_arr == i])[ok_ind]
                else:
                    print("INFO: Find ", ok_count, " small pictures match the requirement: pic_type=", i,
                          " and bool_select=", bool_select)
                    print("INFO: Randomly select ", pic_num_each_comb, " of them...")
                    perm = np.random.permutation(ok_count)
                    ind_perm = perm[:pic_num_each_comb]
                    ok_labels = (key_arr[val_arr == i])[ok_ind][ind_perm]

                result_set = result_set.union(ok_labels.tolist())

        print("INFO: Finally ...... generated a validation set of size ", len(result_set))

        # Copy the Train small picture to a separate folder --- out_dir/JPEGImages
        print("INFO: Copy the validation small pictures to ", out_dir, "/JPEGImages")
        if not os.path.exists("%s/JPEGImages" % out_dir):
            os.makedirs("%s/JPEGImages" % out_dir)

        # Copy the TrainDotted small picture to a separate folder --- out_dir/ValidDotted
        print("INFO: Copy the Dotted validation small pictures to ", out_dir, "/ValidDotted")
        if not os.path.exists("%s/ValidDotted" % out_dir):
            os.makedirs("%s/ValidDotted" % out_dir)

        # Copy corresponding labels to a separate folder --- out_dir/labels
        print("INFO: Copy the validation small picture labels to ", out_dir, "/labels")
        if not os.path.exists("%s/labels" % out_dir):
            os.makedirs("%s/labels" % out_dir)

        with open(out_dir+"/valid_sealion.txt", "wb") as f:
            for item in result_set:
                copyfile(image_dir+"/"+item[:-4]+".JPEG", out_dir+"/JPEGImages/"+item[:-4]+".JPEG")
                copyfile(label_dir+"/"+item, out_dir+"/labels/"+item)
                copyfile(dot_image_dir+"/"+item[:-4]+".JPEG", out_dir+"/ValidDotted/"+item[:-4]+".JPEG")
                f.write(out_dir+"/JPEGImages/"+item[:-4]+".JPEG\n")
            

        # Write label txt file of result_set which points to this new created folder --- out_dir/valid_sealion.txt
        print("INFO: Generate the validation label txt to ", out_dir, "/valid_sealion.txt")

        # Return the pwd of this txt file
        return "%s/valid_sealion.txt" % out_dir

def trigger_validation(darknet_dir, weight_dir, valid_txt_dir, valid_pred_dir):
    if not os.path.exists(darknet_dir) or not os.path.exists(weight_dir) or not os.path.exists(valid_txt_dir):
        print("ERROR: darknet or weight folder you provided does not exist. Exit.")
    if not os.path.exists(valid_pred_dir):
        os.makedirs(valid_pred_dir)

    # Change config file of darknet
    cfg_name_file = darknet_dir + "/cfg/sealion.names"
    cfg_valid_file = darknet_dir + "/cfg/sealion_valid.data"
    cfg_cfg_file = darknet_dir + "/cfg/sealion.2.0.cfg"
    with open(cfg_valid_file, 'wb') as v_f:
        v_f.write("classes = 4\n")
        v_f.write("train = %s\n" % valid_txt_dir)
        v_f.write("names = %s\n" % cfg_name_file)
        v_f.write("valid = %s\n" % valid_txt_dir)
        v_f.write("backup = backup")
    if not os.path.exists(cfg_name_file):
        with open(cfg_name_file, 'wb') as n_f:
            n_f.write("ad_male\nsub_male\njuvenile\nad_female")
        print("INFO: Generated ", cfg_name_file)
    print("INFO: Generated ", cfg_valid_file)

    # For each weight, call ./darknet detector validsealion cfg/sealion.data cfg/sealion.2.0.cfg weight_dir/sealion_800.weights -out valid_pred_dir
    for w in os.listdir(weight_dir):
        print("INFO: Run validation prediction on weight file ", w)
        w_out_dir = "%s/%s" % (valid_pred_dir, w.split('.')[0])
        w_file = "%s/%s" % (weight_dir, w)
        if not os.path.exists(w_out_dir):
            os.makedirs(w_out_dir)
        cmd = "cd %s && ./darknet detector validsealion %s %s %s -out %s" % (darknet_dir, cfg_valid_file, cfg_cfg_file, w_file, w_out_dir)
        print("INFO: Run command --- %s" % cmd)
        os.system(cmd)
    # Parse the log of validation and compare it with true value
    # Draw plots for all weights comparision, write out table /graph
    return



def batch_rename(path):
    for file_name in os.listdir(path):
        os.rename(path + "/" + file_name, path + "/" + file_name[:-3] + "JPEG")


def gen_neg_label(neg_img_dir, valid_txt_dir):
    img_dir = "%s/JPEGImages" % neg_img_dir
    label_dir = "%s/labels" % neg_img_dir
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    if valid_txt_dir != "":
        print("INFO: Will write negative image paths into validation txt...")
        with open(valid_txt_dir, 'a') as v_f:
            for img_name in os.listdir(img_dir):
                with open(label_dir + "/" + img_name[:-4] + "txt", 'wb') as f:
                    f.write("")
                v_f.write("%s/%s\n" % (img_dir, img_name))
    else:
        print("WARN: Will not write validation txt file since you did not provide it.")
        for img_name in os.listdir(img_dir):
            with open(label_dir + "/" + img_name[:-4] + "txt", 'wb') as f:
                f.write("")

        

if __name__ == '__main__':
    # gen_nonzero_label(label_dir, image_dir, out_dir)
    # batch_rename("./data/Yolo_mark_result/JPEGImages")
    # gen_less_female_nonzero_label(label_dir, image_dir, out_dir, 0.05, True)
    # gen_validation_label(label_dir, image_dir, dot_image_dir, out_dir)
    # gen_neg_label(neg_img_dir, valid_txt_dir)
    gen_empty_label("./data/negative_split", "/home/sleepywyn/Dev/GitRepo/Sealion/data/negative_split")
    # trigger_validation(darknet_dir, weight_dir, valid_txt_dir, valid_pred_dir)
    

import os
import operator
import matplotlib.pylab as plt

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
    if not os.path.exists(label_dir):
        print("ERROR: The label folder you provided does not exists. Exit.")
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


def batch_rename(path):
    for file_name in os.listdir(path):
        os.rename(path + "/" + file_name, path + "/" + file_name[:-3] + "JPEG")


if __name__ == '__main__':
    # gen_nonzero_label(label_dir, image_dir, out_dir)
    # batch_rename("./data/Yolo_mark_result/JPEGImages")
    gen_less_female_nonzero_label(label_dir, image_dir, out_dir, 0.05, True)

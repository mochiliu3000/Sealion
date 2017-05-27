import os
import operator
import matplotlib.pylab as plt

label_dir = "C:/Users/IBM_ADMIN/Desktop/Machine Learning/seaLion/data/labels"
image_dir = "C:/Users/IBM_ADMIN/Desktop/Machine Learning/seaLion/data/JPEGImages"
out_dir = "C:/Users/IBM_ADMIN/Desktop/Machine Learning/seaLion/data"

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
			labels = [line.split()[0] for line in open(label_dir + "/" + label_file)]
			num_f_lines = sum([int(i == '3') for i in labels])
			f_dict[label_file] = num_f_lines
		if hist:
			plt.hist(f_dict.values())
			plt.show()
		sorted_f_dict = sorted(f_dict.items(), key=operator.itemgetter(1))
		rm_dict_leng = int(round(len(sorted_f_dict) * threshold))
		rm_dict = sorted_f_dict[-rm_dict_leng:]
		print("Remove these images: ", rm_dict)
		rm_imgs = [i[0] for i in rm_dict]
		with open(out_dir + "/train_new.txt", "wb") as tr_file:
			for label_file in os.listdir(label_dir):
				num_lines = sum(1 for line in open(label_dir + "/" + label_file))
				if num_lines > 0 and label_file not in rm_imgs:
					tr_file.write(image_dir + "/" + label_file[:-3] + "JPEG\n")
						

def batch_rename(path):
	for file_name in os.listdir(path):
		os.rename(path + "/" + file_name, path + "/" + file_name[:-3] + "JPEG")


if __name__ == '__main__':
	# gen_nonzero_label(label_dir, image_dir, out_dir)
	# batch_rename("./data/Yolo_mark_result/JPEGImages")
	gen_less_female_nonzero_label(label_dir, image_dir, out_dir, 0.3, True)
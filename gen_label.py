import os

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

def batch_rename(path):
	for file_name in os.listdir(path):
		os.rename(path + "/" + file_name, path + "/" + file_name[:-3] + "JPEG")


if __name__ == '__main__':
	gen_nonzero_label(label_dir, image_dir, out_dir)
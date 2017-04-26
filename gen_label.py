import os

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
					
gen_nonzero_label(label_dir, image_dir, out_dir)
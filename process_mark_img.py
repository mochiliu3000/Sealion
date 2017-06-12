from skimage import io, transform
import os
import numpy as np

in_dir = "/home/sleepywyn/Dev/GitRepo/Sealion/data/negative_split"
out_dir = "/home/sleepywyn/Dev/GitRepo/Sealion/data/negative_split"
# degrees = [90, 180, 270]
# flips = ["h", "v"]
degrees = [90]
flips = ["h"]

############
## rotate ##
############
def rotate_label(txt_path, out_path, degree):
	with open("%s_r%s.txt" % (out_path, degree), "wb") as wf:
		with open(txt_path) as rf:
			for line in rf:
				args = [float(a) for a in line.split()]
				label = int(args[0])
				args.append(degree)
				x, y, width, height = rotate_line(*args[1:])
				wf.write("%s %s %s %s %s\n" % (label, x, y, width, height))

def rotate_line(x, y, width, height, degree):
	if degree == 90:
		return y, 1-x, height, width
	if degree == 180:
		return 1-x, 1-y, width, height
	if degree == 270:
		return 1-y, x, height, width

def rotate_img(img, out_path, degree):	
	img_r = transform.rotate(img, degree, resize=True)
	io.imsave("%s_r%s.jpg" % (out_path, degree), img_r)
	return
	
##########
## flip ##
##########
def flip_label(txt_path, out_path, direction):
	with open("%s_%s.txt" % (out_path, direction), "wb") as wf:
		with open(txt_path) as rf:
			for line in rf:
				args = [float(a) for a in line.split()]
				label = int(args[0])
				args.append(direction)
				x, y, width, height = flip_line(*args[1:])
				wf.write("%s %s %s %s %s\n" % (label, x, y, width, height))
				
def flip_line(x, y, width, height, direction):
    if direction == "h":
        return 1-x, y, width, height
    if direction == "v":
        return x, 1-y, width, height

def flip_img(img, out_path, direction):
	if direction == "h":
		img_h = np.fliplr(img)
	if direction == "v":
		img_h = np.flipud(img)
	io.imsave("%s_%s.jpg" % (out_path, direction), img_h)
	return
	

def process_mark_img(in_dir, out_dir, degrees, flips):
	file_names = os.listdir(in_dir)
	jpg_files = [file_names[i] for i, f in enumerate(file_names) if ".jpg" in f]
	txt_files = [file_names[i] for i, f in enumerate(file_names) if ".txt" in f]
	for jpg_f in jpg_files:
		if (jpg_f[:-4] + ".txt") not in txt_files:
			print("WARNING: The corresponding txt file for %s does not exist, skip this image..." % jpg_f)
			continue
		else:
			img_path = "%s/%s" % (in_dir, jpg_f)
			txt_path = "%s/%s.txt" % (in_dir, jpg_f[:-4])
			out_path = "%s/%s" % (out_dir, jpg_f[:-4])
			img = io.imread(img_path)
			for degree in degrees:
				rotate_img(img, out_path, degree)
				rotate_label(txt_path, out_path, degree)
			for flip in flips:
				flip_img(img, out_path, flip)
				flip_label(txt_path, out_path, flip)
			

if __name__ == '__main__':
	process_mark_img(in_dir, out_dir, degrees, flips)
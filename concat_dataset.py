import os
import subprocess
import random
import dircache

# number = 0
# root = '/u/ramrao/data/FRCNN/datasets/test'
# classes = dircache.listdir(root)
# for values in classes:
# 	dir = os.path.join(root,values)
# 	filename = random.choice(dircache.listdir(dir))
# 	path_val = os.path.join(dir, filename)
# 	subprocess.call(["convert", path_val, "-resize", "46x56!", path_val])
# 	images = dircache.listdir(os.path.join(root,values))
# 	for i in images:
# 		os.chdir('/u/ramrao/siamese/datasets/test/label1')
# 		subprocess.call(["convert", path_val, "-resize", "46x56!", path_val])
# 		subprocess.call(["convert", os.path.join(root,values,i), "-resize", "46x56!", os.path.join(root, values, i)])
# 		subprocess.call(["convert", os.path.join(root,values, i), path_val ,"+append", str(number) + "_" + i])
# 		number += 1

number = 0
main_root = '/u/ramrao/data/FRCNN/datasets/test'
list_val = ['brad_pitt', 'cristiano_ronaldo', 'david_beckham', 'emma_watson', 'hillary_clinton', 'keira_knightley']
root = '/u/ramrao/data/FRCNN/datasets/test/barack_obama'
filename = random.choice(dircache.listdir(root))
path_val = os.path.join(root, filename)
subprocess.call(["convert", path_val, "-resize", "46x56!", path_val])
for i in list_val:
	for images in dircache.listdir(os.path.join(main_root,i)):
		os.chdir('/u/ramrao/siamese/datasets/test/label1')
		subprocess.call(["convert", path_val, "-resize", "46x56!", path_val])
		subprocess.call(["convert", os.path.join(main_root,i,images), "-resize", "46x56!", os.path.join(main_root, i, images)])
		subprocess.call(["convert", os.path.join(main_root,i, images), path_val ,"+append", str(number) + "_" + i])
		number += 1



print("all done")

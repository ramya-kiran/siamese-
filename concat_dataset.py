import os
import subprocess
import random
import dircache
import copy
import numpy as np

# root = '/u/ramrao/siamese/shapes/test/'
# classes = dircache.listdir(root)
# for values in classes:
# 	dir = os.path.join(root,values)
# 	all_images = dircache.listdir(dir)
# 	print(all_images)
# 	#filename = random.choice(dircache.listdir(dir))
# 	for j in range(5):
# 		print(j)
# 		print(all_images)
# 		filename = all_images[j]
# 		path_val = os.path.join(dir, filename)
# 		#subprocess.call(["convert", path_val, "-resize", "46x56!", path_val])
# 		#images = dircache.listdir(os.path.join(root,values))
# 		temp_images = copy.deepcopy(all_images)
# 		temp_images.remove(filename)
# 		for i in temp_images:
# 			os.chdir('/u/ramrao/siamese/datasets/test/label0')
# 			subprocess.call(["convert", path_val, "-resize", "46x56!", path_val])
# 			subprocess.call(["convert", os.path.join(root,values,i), "-resize", "46x56!", os.path.join(root, values, i)])
# 			subprocess.call(["convert", os.path.join(root,values, i), path_val ,"+append", filename[0:(len(filename)-4)] + "_" + i])
		

root = '/u/ramrao/siamese/shapes/test/'
classes = dircache.listdir(root)
index_val = 0
for values in classes:
	dir_val = os.path.join(root,values)
	all_images = dircache.listdir(dir_val)
	print(all_images)
	#filename = random.choice(dircache.listdir(dir))
	random.shuffle(all_images)
	for j in range(4):
		print(j)
		print(all_images)
		filename = all_images[j]
		path_val = os.path.join(dir_val, filename)
		#subprocess.call(["convert", path_val, "-resize", "46x56!", path_val])
		#images = dircache.listdir(os.path.join(root,values))
		temp_images = dircache.listdir(os.path.join(root, classes[abs(1 - index_val)]))
		cross_path = os.path.join(root, classes[abs(1 - index_val)])
		for i in temp_images:
			os.chdir('/u/ramrao/siamese/datasets/test/label1')
                        subprocess.call(["convert", path_val, "-resize", "46x56!", path_val])
			subprocess.call(["convert", os.path.join(cross_path,i), "-resize", "46x56!", os.path.join(cross_path, i)])
			subprocess.call(["convert", os.path.join(cross_path, i), path_val ,"+append", filename[0:(len(filename)-4)] + "_" + i])

	index_val += 1  

print("all done")

# This code splits the data into train, eval, and test.
# The original data folder should have n folders where
# each folder contains images for each of the n classes

import os
import sys
import random

base_folder = "data_new"
target_folder = "data_norm"

os.mkdir(target_folder)

for folder in os.listdir(base_folder):

	source = base_folder + "/" + folder
	curr_folder = target_folder + "/" + folder
	os.mkdir(curr_folder)

	os.mkdir(curr_folder + "/train")
	os.mkdir(curr_folder + "/eval")
	os.mkdir(curr_folder + "/test")

	all_files = os.listdir(source)
	random.shuffle(all_files)

	n = len(all_files)

	idx1 = int(n*0.8)
	idx2 = int(n*0.9)

	for file in all_files[:idx1]:
		os.system("mv %s %s"%(source+"/"+file, curr_folder+"/train/"))

	for file in all_files[idx1:idx2]:
		os.system("mv %s %s"%(source+"/"+file, curr_folder+"/eval/"))

	for file in all_files[idx2:]:
		os.system("mv %s %s"%(source+"/"+file, curr_folder+"/test/"))
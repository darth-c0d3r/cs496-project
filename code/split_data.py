import os
import random
import shutil

base = "data_new"
classes = os.listdir(base)
folders = ["train", "test", "eval"]

for cl in classes:

	files = os.listdir(base+"/"+cl)
	random.shuffle(files)

	for fldr in folders:
		if fldr not in os.listdir(base+"/"+cl):
			os.mkdir(base+"/"+cl+"/"+fldr)

	probe1 = int(len(files)*0.8)
	probe2 = int(len(files)*0.9)

	for idx in range(probe1):
		shutil.move(base+"/"+cl+"/"+files[idx], base+"/"+cl+"/train/"+files[idx])

	print("Moved %s %s"%(cl, folders[0]))

	for idx in range(probe1, probe2):
		shutil.move(base+"/"+cl+"/"+files[idx], base+"/"+cl+"/eval/"+files[idx])

	print("Moved %s %s"%(cl, folders[1]))

	for idx in range(probe2,len(files)):
		shutil.move(base+"/"+cl+"/"+files[idx], base+"/"+cl+"/test/"+files[idx])

	print("Moved %s %s"%(cl, folders[2]))

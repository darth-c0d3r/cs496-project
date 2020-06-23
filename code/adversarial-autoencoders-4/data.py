from PIL import Image
import os
import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import Dataset


#---------------------- DUMMY-DATASET ----------------------#

class DummyDataset(Dataset):

	def __init__(self, size, shape, num_classes=2):
		"""
		size is the number of samples per class
		shape is the shape of each sample

		size =:= 100
		shape =:= [3,256,256]
		num_classes =:= 4

		"""
		self.size = size
		self.shape = shape
		self.num_classes = num_classes

	def __len__(self):
		return self.num_classes*self.size

	def __getitem__(self, idx):

		target = idx % self.num_classes
		# normalize dataset externally; don't worry about it now

		data = torch.randn(self.shape)*0.1 + target

		return data, torch.tensor(target)

def getDummyDataset(num_classes = 2):

	size = [500, 100]
	shape = [3,128,128]
	
	# size = [5,1]
	# shape = [3]
	# num_classes = 2

	return {"train":DummyDataset(size[0], shape, num_classes), "eval":DummyDataset(size[1], shape, num_classes)}

def getDummyClassDataset(num_classes = 2):

	size = [500, 100]
	shape = [3,128,128]
	
	dataset = []
	for _ in range(num_classes):
		dataset.append(getDummyDataset(1))

	return dataset





# For some dummy testing

# data = getDummyDataset()
# for i in range(len(data["train"])):
# 	print(data["train"][i])
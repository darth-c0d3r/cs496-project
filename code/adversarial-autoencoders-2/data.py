from PIL import Image
import os
import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import random


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

def getDummyDataset():

	size = [500, 100]
	shape = [3,128,128]
	num_classes = 2

	# size = [5,1]
	# shape = [3]
	# num_classes = 2

	return {"train":DummyDataset(size[0], shape, num_classes), "eval":DummyDataset(size[1], shape, num_classes)}

#---------------------- BREAKHIS-DATASET ----------------------#

class BreakhisDataset(Dataset):
	
	def __init__(self, folder, noise=False, p=0):

		super(BreakhisDataset, self).__init__()

		self.shape = [3,460,700]
		self.num_classes = 2

		self.base = "../data_new"

		self.neg = os.listdir(self.base + "/Benign/" + folder) # 0
		self.pos = os.listdir(self.base + "/Malign/" + folder) # 1

		self.neg = [(self.base+"/Benign/"+folder+"/"+file, 0) for file in self.neg]
		self.pos = [(self.base+"/Malign/"+folder+"/"+file, 1) for file in self.pos]

		self.all = self.neg + self.pos
		p = (p/100.0)*len(self.all)

		if noise is True:
			choices = random.sample(range(len(self.all)), k=int(p))
			for idx in choices:
				self.all[idx] = (self.all[idx][0], (self.all[idx][1]+1)%2)

		print("About Dataset [%s]"%(folder))
		print("Benign: %d"%(len(self.neg)))
		print("Malign: %d"%(len(self.pos)))
		print("Total: %d"%(len(self.all)))
		print("Noise: %04f"%((p*100.0)/len(self.all)))
		print("Shape: "+str(self.shape))
		print()

	def __len__(self):
		return len(self.all)

	def __getitem__(self, idx):

		img = Image.open(self.all[idx][0])
		target = self.all[idx][1]

		img = img.resize(self.shape[1:][::-1])
		trans = torchvision.transforms.ToTensor()

		return ((2.*trans(img))-1., torch.tensor(target).long())


def getBreakhisDataset():

	return{ "train":BreakhisDataset("train", True, 10),
			"eval":BreakhisDataset("eval"),
			"test":BreakhisDataset("test")}


# For some dummy testing

# data = getDummyDataset()
# for i in range(len(data["train"])):
# 	print(data["train"][i])

# data = getBreakhisDataset()
# print(data["train"][0][0].shape)
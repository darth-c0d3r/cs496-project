from PIL import Image
import os
import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import Dataset


#---------------------- DUMMY-DATASET ----------------------#

class DummyBinaryDataset(Dataset):

	def __init__(self, size, shape):
		"""
		size is the number of samples per class
		shape is the shape of each sample
		"""
		self.size = size
		self.shape = shape

	def __len__(self):
		return 2*self.size

	def __getitem__(self, idx):

		target = idx % 2
		data = torch.randn(self.shape)*0.1 + 2*target - 1

		return data, torch.tensor([target])

def getDummyBinaryDataset():

	size = [500, 100]
	shape = [3,128,128]
	# shape = [256]

	return {"train":DummyBinaryDataset(size[0], shape), "eval":DummyBinaryDataset(size[1], shape)}


#---------------------- BREAKHIST-DATASET ----------------------#

class BreakHistDataset(Dataset):

	def __init__(self, folder):
		"""
		folder is one of "train", "test", "eval"
		"""

		pos = "data_norm/Malignant/"+folder+"/"
		neg = "data_norm/Benign/"+folder+"/"

		files_pos = os.listdir(pos)
		files_neg = os.listdir(neg)

		self.lables = [1 for _ in range(len(files_pos))] + [0 for _ in range(len(files_neg))]
		files_pos = [pos + name for name in files_pos]
		files_neg = [neg + name for name in files_neg]

		self.files = files_pos + files_neg

		self.shape = [3,460,700]

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		img = Image.open(self.files[idx])
		img = img.resize((460, 700))
		trans = torchvision.transforms.ToTensor()
		return ((2.*trans(img))-1., torch.tensor(self.lables[idx]).long())


def getBreakHistDataset():
	return {"train":BreakHistDataset("train"),
			"test":BreakHistDataset("test"),
			"eval":BreakHistDataset("eval")}
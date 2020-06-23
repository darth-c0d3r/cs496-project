import torch
import torch.nn as nn
import torch.optim as optim

from model import EncoderNetwork
from model import FullyConnectedNetwork
from data import getDummyBinaryDataset
from data import getBreakHistDataset
from evaluate import evaluate_conv
from util import *

def train(enc, clsfr, dataset, device):
	"""
	enc and clsfr are networks
	dataset["train"] and dataset["eval"] can be used
	device is either CPU or GPU
	"""

	# hyperparameters
	epochs = 25
	batch_size = 50

	# Loss Functions
	ClsfrLoss = nn.CrossEntropyLoss().to(device)

	# Optimizers
	opt = optim.Adam(list(enc.parameters()) + list(clsfr.parameters()),
					 lr=0.002, betas=(0.5, 0.999))

	# iterate for epochs
	for epoch in range(1, epochs):

		enc.train()
		clsfr.train()

		# get the data loader
		dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)

		num_rounds = (len(dataloader)+1)//batch_size

		total = 0
		num_steps = 0
		epoch_loss = 0.
		epoch_correct = 0

		for data, target in dataloader:

			data = data.to(device)
			target = target.to(device)

			# data.shape = [n,3,l,b]
			# target.shape = [n,1]
			target = target.squeeze(-1)

			# TRAIN

			# set gradients to zero
			opt.zero_grad()
			enc.zero_grad()
			clsfr.zero_grad()

			# get loss values and update params

			out = clsfr(enc(data))
			loss = ClsfrLoss(out, target)
			loss.backward()
			opt.step()

			labels = torch.argmax(out, 1)
			correct = torch.sum(labels == target)

			epoch_loss = (epoch_loss*num_steps + loss)/(num_steps + 1)
			total += len(data)
			num_steps += 1
			epoch_correct += correct

			print("Epoch: %004d \t Loss: %004f \t Correct: %004d/%004d (%004f)"%\
				(epoch, loss, correct, len(data), float(correct)/float(len(data))))

		print("Epoch [Train]: %004d \t Loss: %004f \t Correct: %004d/%004d (%004f)"%\
				(epoch, epoch_loss, epoch_correct, total, float(epoch_correct)/float(total)))
		evaluate_conv(enc, clsfr, dataset, device, epoch)

		print()



def main():

	device = get_device(True)
	# dataset = getDummyBinaryDataset()
	dataset = getBreakHistDataset()

	conv = [3,4,8,16]
	fc = [128,64,32,16]
	shape = dataset["train"].shape

	enc = EncoderNetwork(conv, shape).to(device)
	clsfr = FullyConnectedNetwork([enc.size]+fc, 2).to(device)

	train(enc, clsfr, dataset, device)

main()
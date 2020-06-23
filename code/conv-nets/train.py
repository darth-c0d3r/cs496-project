import torch
import torch.nn as nn
import torch.optim as optim

from util import *
from data import getDummyDataset
from data import getBreakhisDataset
from model import *

def train(enc, disc, dataset, device):

	"""
	enc and disc are networks
	dataset["train"] and dataset["eval"] can be used
	device is either CPU or GPU
	"""

	# hyperparameters
	epochs = 100
	batch_size = 50

	# Loss Functions
	DiscLoss = nn.CrossEntropyLoss().to(device)

	# Optimizers
	main_optim = optim.Adam(list(enc.parameters()) + list(disc.parameters()), lr=0.0002, betas=(0.5, 0.999))

	# iterate for epochs
	for epoch in range(1, epochs+1):

		enc.train()
		disc.train()

		# get the data loader
		dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)

		loss_epoch = 0.
		correct_epoch = 0

		for data, target in dataloader:

			# data.shape = [n,3,l,b]
			# target.shape = [n]

			data = data.to(device)
			target = target.to(device)

			# TRAIN DISCRIMINATOR

			# set gradients to zero
			main_optim.zero_grad()
			enc.zero_grad()
			disc.zero_grad()

			# get output of discriminator
			hidden = enc(data).view(data.shape[0], -1)
			out = disc(hidden)

			# calculate loss and update params
			loss = DiscLoss(out, target)
			loss.backward()
			main_optim.step()

			# get accuracy and loss_epoch
			correct = torch.sum(target == torch.argmax(out, 1))

			loss_epoch += len(data)*loss
			correct_epoch += correct

			print(".", end="")

		loss_epoch = loss_epoch/len(dataset["train"])
		
		# Pretty Printing
		print("")
		print("Epoch %04d/%04d : Loss : %06f \t Accuracy : %04d/%04d (%06f)"%\
			(epoch, epochs, loss_epoch, correct_epoch, len(dataset["train"]), correct_epoch*100.0/float(len(dataset["train"]))))
		eval_model(enc, disc, dataset, device, "eval")
		print()

	return enc, disc

def main():

	device = get_device(True)
	dataset = getBreakhisDataset()

	conv = [3,4,8,16]
	fc = [32,16,2]
	shape = dataset["train"].shape


	enc = EncoderNetwork(conv, shape).to(device)
	disc = FullyConnectedNetwork(fc, enc.size).to(device)

	enc, disc = train(enc, disc, dataset, device)
	save_model(enc, "enc5.pt")
	save_model(disc, "disc5.pt")

main()
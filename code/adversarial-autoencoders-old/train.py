import torch
import torch.nn as nn
import torch.optim as optim

from model import EncoderNetwork
from model import DecoderNetwork
from model import FullyConnectedNetwork
from data import getDummyBinaryDataset
from util import *

def train(enc, dec, disc, dataset, device):
	"""
	enc, dec, and disc are networks
	dataset["train"] and dataset["eval"] can be used
	device is either CPU or GPU
	"""

	# hyperparameters
	epochs = 25
	batch_size = 50

	# Loss Functions
	DiscLoss = nn.CrossEntropyLoss().to(device)
	DecoderLoss = nn.MSELoss().to(device)

	# Optimizers
	disc_optim = optim.Adam(disc.parameters(), lr=0.002, betas=(0.5, 0.999))

	main_optim = optim.Adam(list(enc.parameters()) + list(dec.parameters()) + 
					list(disc.parameters()), lr=0.002, betas=(0.5, 0.999))

	# iterate for epochs
	for epoch in range(1, epochs):

		enc.train()
		dec.train()
		disc.train()

		# get the data loader
		dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)

		for data, target in dataloader:

			# data.shape = [n,3,l,b]
			# target.shape = [n,1]

			# TRAIN DISCRIMINATOR

			# set gradients to zero
			disc_optim.zero_grad()
			disc.zero_grad()

			# get hidden, prior, and one-hot vectors
			hidden = enc(data).view(data.shape[0], -1)
			prior = torch.randn(hidden.shape)*0.1 + 2*target - 1
			append = get_onehot(target)

			# append one-hot vector to both hidden and prior
			hidden = torch.cat([hidden, append], 1)
			prior = torch.cat([prior, append], 1)

			# concatenate the two to get new X and Y
			X_disc = torch.cat([hidden, prior], 0)
			Y_hidden = torch.zeros((len(hidden))).long().to(device)
			Y_prior = torch.ones((len(prior))).long().to(device)
			Y_disc = torch.cat([Y_hidden, Y_prior], 0)

			# get loss values corresponding to hidden
			out = disc(X_disc)
			loss = DiscLoss(out, Y_disc)
			loss.backward()
			disc_optim.step()
			
			# TRAIN ENCODER AND DECODER

			# set gradients to zero
			main_optim.zero_grad()
			enc.zero_grad()
			dec.zero_grad()
			disc.zero_grad()

			# get the hidden and one-hot vectors
			hidden = enc(data)
			data_ = dec(hidden)
			hidden = hidden.view(data.shape[0], -1)
			append = get_onehot(target)

			hidden = torch.cat([hidden, append], 1)
			Y_hidden = torch.zeros((len(hidden))).long().to(device)
			out = disc(hidden)

			loss = DecoderLoss(data, data_) - DiscLoss(out, Y_hidden)
			loss.backward()
			main_optim.step()

			print(loss)


def main():

	device = get_device(True)
	dataset = getDummyBinaryDataset()

	conv = [3,4,8,16]
	deconv = conv[::-1]
	fc = [128,64,32,16]
	shape = dataset["train"].shape

	enc = EncoderNetwork(conv, shape).to(device)
	dec = DecoderNetwork(deconv, enc.output_padding).to(device)
	disc = FullyConnectedNetwork([enc.size+2]+fc, 2).to(device) # +2 for appending class

	train(enc, dec, disc, dataset, device)

main()
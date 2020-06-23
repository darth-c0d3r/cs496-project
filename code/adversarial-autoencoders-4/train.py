import torch
import torch.nn as nn
import torch.optim as optim

from util import *
from data import getDummyDataset
from data import getDummyClassDataset
from model import *

def train(enc, dec, disc, dataset, device, num_classes):

	"""
	enc, dec, and disc are lists of networks
	dataset is a list of datasets
	dataset[i]["train"] and dataset[i]["eval"] can be used
	device is either CPU or GPU
	"""

	# hyperparameters
	epochs = 25
	batch_size = 50
	lamda = 1.0

	# Loss Functions
	DiscLoss = nn.CrossEntropyLoss().to(device)
	RecreationLoss = nn.MSELoss().to(device)

	# iterate for epochs
	for epoch in range(1, epochs):

		# Optimizers
		disc_optim = [optim.Adam(disc[cl].parameters(), lr=0.0002, betas=(0.5, 0.999)) for cl in range(num_classes)]
		main_optim = [optim.Adam(list(enc[cl].parameters()) + list(dec[cl].parameters()),\
						 lr=0.0002, betas=(0.5, 0.999)) for cl in range(num_classes)]

		# iterate over all classes (all networks)
		for cl in range(num_classes):
		

			# set the flags to train
			enc[cl].train()
			dec[cl].train()
			disc[cl].train()

			# get the data loader
			dataloader = torch.utils.data.DataLoader(dataset[cl]["train"], batch_size=batch_size, shuffle=True)

			# initialize loss values
			loss_disc_1 = 0.
			loss_disc_2 = 0.
			loss_rec = 0.
			correct_1 = 0
			correct_2 = 0

			# iterate over mini batches
			for data, target in dataloader:

				# put data and target to device
				data = data.to(device)
				target = target.to(device)

				# data.shape = [n,3,l,b]
				# target.shape = [n]

				# TRAIN DISCRIMINATOR
				# 0 means it is from encoder
				# 1 means it is from prior

				# set gradients to zero
				disc_optim[cl].zero_grad()
				enc[cl].zero_grad()
				disc[cl].zero_grad()

				# get hidden and sample prior
				hidden = enc[cl](data)
				prior = torch.randn(hidden.shape)

				# concatenate to get X and Y
				X = torch.cat([hidden, prior])
				Y = torch.cat([torch.zeros(hidden.shape[0]), torch.ones(hidden.shape[0])]).long().to(device)

				# get output of discriminator
				out = disc[cl](X.view(X.shape[0], -1))

				# calculate loss and update params
				loss1 = DiscLoss(out, Y)
				loss1.backward()
				disc_optim[cl].step()

				# get accuracy and loss_epoch
				correct = torch.sum(Y == torch.argmax(out, 1))
				loss_disc_1 += len(X)*loss1
				correct_1 += correct

				# TRAIN ENCODER AND DECODER
				# 0 means it is from encoder
				# 1 means it is from prior

				# set gradients to zero
				main_optim[cl].zero_grad()
				enc[cl].zero_grad()
				dec[cl].zero_grad()
				disc[cl].zero_grad()

				# get hidden and sample prior
				hidden = enc[cl](data)
				prior = torch.randn(hidden.shape)

				# add reconstruction error to loss
				data_ = dec[cl](hidden)
				loss2 = RecreationLoss(data, data_)
				loss_rec += len(data)*loss2

				# concatenate to get X and Y
				X = torch.cat([hidden, prior])
				Y = torch.cat([torch.zeros(hidden.shape[0]), torch.ones(hidden.shape[0])]).long().to(device)

				# get output of discriminator and update it
				out = disc[cl](X.view(X.shape[0], -1))

				# calculate loss and update params
				loss3 = DiscLoss(out, Y)
				loss_disc_2 += len(X)*loss3

				loss = loss2 - lamda*loss3
				loss.backward()
				main_optim[cl].step()

				# get accuracy and loss_epoch
				correct = torch.sum(Y == torch.argmax(out, 1))
				correct_2 += correct


			loss_disc_1 = loss_disc_1/(2*len(dataset[cl]["train"]))
			loss_disc_2 = loss_disc_2/(2*len(dataset[cl]["train"]))
			loss_rec = loss_rec/len(dataset[cl]["train"])
			
			# Pretty Printing
			print("Class %d"%cl)
			print("[Disc1] Epoch %04d/%04d : Loss : %06f \t Accuracy : %04d/%04d (%06f)"%\
				(epoch, epochs, loss_disc_1, correct_1, 2*len(dataset[cl]["train"]), correct_1*50.0/float(len(dataset[cl]["train"]))))
			print("[Disc2] Epoch %04d/%04d : Loss : %06f \t Accuracy : %04d/%04d (%06f)"%\
				(epoch, epochs, loss_disc_2, correct_2, 2*len(dataset[cl]["train"]), correct_2*50.0/float(len(dataset[cl]["train"]))))
			print("[Dec] Epoch %04d/%04d : Loss : %06f"%\
				(epoch, epochs, loss_rec))
			print()

			# eval_model(enc, disc, dataset, device)
			# print()

	eval_model(enc, dataset, device, num_classes, "train")

def main():

	num_classes = 2

	device = get_device(True)
	dataset = getDummyClassDataset(num_classes)

	conv = [3,4,8,16]
	fc = [128,64,32,16,2]
	shape = dataset[0]["train"].shape

	enc = []
	dec = []
	disc = []

	for _ in range(num_classes):
		enc.append(EncoderNetwork(conv, shape).to(device))
		dec.append(DecoderNetwork(conv[::-1], enc[-1].out_p).to(device))
		disc.append(FullyConnectedNetwork(fc, enc[-1].size).to(device))

	train(enc, dec, disc, dataset, device, num_classes)

main()
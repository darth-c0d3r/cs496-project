import torch
import torch.nn as nn
import torch.optim as optim

from util import *
from data import getDummyDataset
from model import *

def train(enc, dec, clsfr, dataset, device):

	"""
	enc, dec, and clsfr are networks
	dataset["train"] and dataset["eval"] can be used
	device is either CPU or GPU
	"""

	# hyperparameters
	epochs = 25
	batch_size = 50
	lamda = 1.0

	# Loss Functions
	ClsfrLoss = nn.CrossEntropyLoss().to(device)
	RecreationLoss = nn.MSELoss().to(device)

	# Optimizers
	opt = optim.Adam(list(enc.parameters()) + list(dec.parameters()) + list(clsfr.parameters()), lr=0.0002, betas=(0.5, 0.999))

	# iterate for epochs
	for epoch in range(1, epochs):

		# set the flags to train
		enc.train()
		dec.train()
		clsfr.train()

		# get the data loader
		dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)

		# initialize loss values
		loss_clsfr = 0.
		loss_rec = 0.
		correct_epoch = 0

		# iterate over mini batches
		for data, target in dataloader:

			# put data and target to device
			data = data.to(device)
			target = target.to(device)

			# data.shape = [n,3,l,b]
			# target.shape = [n]

			# TRAIN

			# set gradients to zero
			opt.zero_grad()
			enc.zero_grad()
			dec.zero_grad()
			clsfr.zero_grad()

			# get hidden
			hidden = enc(data)
			
			# add reconstruction error to loss
			data_ = dec(hidden)
			loss1 = RecreationLoss(data, data_)
			loss_rec += len(data)*loss1

			# get output of classifier and update it
			out = clsfr(hidden.view(hidden.shape[0], -1))

			# calculate Clasfier Loss
			loss2 = ClsfrLoss(out, target)
			loss_clsfr += len(data)*loss2

			# add losses and update params
			loss = loss1 + lamda*loss2
			loss.backward()
			opt.step()

			# get accuracy and loss_epoch
			correct = torch.sum(target == torch.argmax(out, 1))
			correct_epoch += correct


		loss_clsfr = loss_clsfr/len(dataset["train"])
		loss_rec = loss_rec/len(dataset["train"])
		
		# Pretty Printing
		print("[Clsfr] Epoch %04d/%04d : Loss : %06f \t Accuracy : %04d/%04d (%06f)"%\
			(epoch, epochs, loss_clsfr, correct_epoch, len(dataset["train"]), correct_epoch*100.0/float(len(dataset["train"]))))
		print("[Dec] Epoch %04d/%04d : Loss : %06f"%\
			(epoch, epochs, loss_rec))
		print()

		# eval_model(enc, disc, dataset, device)
		# print()

def main():

	device = get_device(True)
	dataset = getDummyDataset()

	conv = [3,4,8,16]
	fc = [128,64,32,16,2]
	shape = dataset["train"].shape

	enc = EncoderNetwork(conv, shape).to(device)
	dec = DecoderNetwork(conv[::-1], enc.out_p).to(device)
	clsfr = FullyConnectedNetwork(fc, enc.size).to(device)

	train(enc, dec, clsfr, dataset, device)

main()
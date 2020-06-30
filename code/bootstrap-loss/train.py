import torch
import torch.nn as nn
import torch.optim as optim

from util import *
from data import getDummyDataset
from data import getBreakhisDataset
from model import *

def train(enc, dec, disc, clsfr, dataset, device):

	"""
	enc, dec, disc, and clsfr are networks
	dataset["train"] and dataset["eval"] can be used
	device is either CPU or GPU
	"""

	# hyperparameters
	epochs = 100
	batch_size = 50
	lamda1 = 1.0 # classifier
	lamda2 = 1.0 # discriminator
	variance = 0.1

	weighted = True # weighted updates for clsfr
	beta = 1e-4 # for multiplication in gaussian num
	w_n = True # weight normalization

	# Loss Functions
	DiscLoss_1 = nn.CrossEntropyLoss().to(device)
	DiscLoss_2 = None # placeholder
	RecreationLoss = None # placeholder
	if weighted is True:
		DiscLoss_2 = nn.CrossEntropyLoss(reduction='mean').to(device)
		RecreationLoss = nn.MSELoss(reduction='none').to(device)
	else:
		DiscLoss_2 = nn.CrossEntropyLoss(reduction='mean').to(device)
		RecreationLoss = nn.MSELoss(reduction='mean').to(device)

	# Optimizers
	disc_optim = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))
	main_optim = optim.Adam(list(enc.parameters()) + list(dec.parameters()) + list(clsfr.parameters()), lr=0.0002, betas=(0.5, 0.999))

	# iterate for epochs
	for epoch in range(1, epochs+1):

		# set the flags to train
		enc.train()
		dec.train()
		disc.train()
		clsfr.train()

		# get the data loader
		dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)

		# initialize loss values
		loss_disc_1 = 0.
		loss_disc_2 = 0.
		loss_clsfr = 0.
		loss_rec = 0.
		correct_1 = 0 # disc_1
		correct_2 = 0 # disc_2
		correct_3 = 0 # clsfr
		num_pts = 0

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
			disc_optim.zero_grad()
			enc.zero_grad()
			disc.zero_grad()

			# get hidden and reshape
			hidden = enc(data).view(data.shape[0], -1)

			# sample prior according to target
			prior = (torch.randn(hidden.shape)*variance).to(device)
			prior = prior + 2*target.repeat(prior.shape[1], 1).to(device).T - 1 # just add the class index to mean

			# concatenate to get X and Y
			X = torch.cat([hidden, prior])
			Y = torch.cat([torch.zeros(hidden.shape[0]), torch.ones(hidden.shape[0])]).long().to(device)

			# update X according to the target
			# append the one-hot vector of target

			# calculate the one-hot vector
			idx = torch.Tensor([[i,target[i]] for i in range(len(target))]).long()
			OH = torch.zeros((len(target), dataset["train"].num_classes)).to(device)
			OH[idx[:,0], idx[:,1]] = 1
			OH = torch.cat([OH,OH])

			# append to X
			X = torch.cat([X,OH], 1)

			# get output of discriminator
			out = disc(X)

			# calculate loss and update params
			loss1 = DiscLoss_1(out, Y)
			loss1.backward()
			disc_optim.step()

			# get accuracy and loss_disc_1
			correct = torch.sum(Y == torch.argmax(out, 1))
			loss_disc_1 += len(X)*loss1
			correct_1 += correct

			# TRAIN ENCODER, DECODER, CLASSIFIER
			# 0 means it is from encoder
			# 1 means it is from prior

			# set gradients to zero
			main_optim.zero_grad()
			enc.zero_grad()
			dec.zero_grad()
			disc.zero_grad()

			# get hidden
			hidden = enc(data)

			# getting weights
			weights = hidden.view(hidden.shape[0], -1)
			weights = weights - (2*target.repeat(prior.shape[1], 1).to(device).T - 1)
			weights = beta*weights*weights
			weights = torch.sum(weights, 1)
			weights = torch.exp(-1*weights)

			# print(weights)

			if w_n is True:
				weights = (weights / torch.max(weights).float())

			# print(weights)

			# print(weights)

			# add reconstruction error to loss
			data_ = dec(hidden)
			loss2 = RecreationLoss(data, data_)
			# multiply by weights (if reqd)
			if weighted is True:
				loss2 = loss2.view(loss2.shape[0], -1)
				loss2 = torch.mean(loss2, 1)
				loss2 = torch.mean(loss2*weights)
			loss_rec += len(data)*loss2

			# reshape hidden
			hidden = hidden.view(data.shape[0], -1)

			# get output of classifier and calculate loss
			out1 = clsfr(hidden)
			loss3 = BootstrapLoss(out1, target, weights, device)
			loss_clsfr += len(hidden)*loss3

			# get accuracy of classifier
			correct = torch.sum(target == torch.argmax(out1, 1))
			correct_3 += correct

			# sample prior according to target
			prior = (torch.randn(hidden.shape)*variance).to(device)
			prior = prior + 2*target.repeat(prior.shape[1], 1).to(device).T - 1 # just add the class index to mean

			# concatenate to get X and Y
			X = torch.cat([hidden, prior])
			Y = torch.cat([torch.zeros(hidden.shape[0]), torch.ones(hidden.shape[0])]).long().to(device)

			# update Y according to the target
			# append the one-hot vector of target

			# # calculate the one-hot vector (NO NEED TO DO AGAIN)
			# idx = torch.Tensor([[i,target[i]] for i in range(len(target))]).long()
			# OH = torch.zeros((len(target), dataset["train"].num_classes))
			# OH[idx[:,0], idx[:,1]] = 1
			# OH = torch.cat([OH,OH])

			# append to X
			X = torch.cat([X,OH], 1)

			# get output of discriminator
			out2 = disc(X)

			# calculate disc loss
			loss4 = DiscLoss_2(out2, Y)
			loss_disc_2 += len(X)*loss4


			loss = loss2 + lamda1*loss3 - lamda2*loss4
			loss.backward()
			main_optim.step()

			# get accuracy and loss_disc_2
			correct = torch.sum(Y == torch.argmax(out2, 1))
			correct_2 += correct

		loss_disc_1 = loss_disc_1/(2*len(dataset["train"]))
		loss_disc_2 = loss_disc_2/(2*len(dataset["train"]))
		loss_clsfr = loss_clsfr/(len(dataset["train"]))
		loss_rec = loss_rec/len(dataset["train"])
		
		# Pretty Printing
		print("[Disc1] Epoch %04d/%04d : Loss : %06f \t Accuracy : %04d/%04d (%06f)"%\
			(epoch, epochs, loss_disc_1, correct_1, 2*len(dataset["train"]), correct_1*50.0/float(len(dataset["train"]))))
		print("[Disc2] Epoch %04d/%04d : Loss : %06f \t Accuracy : %04d/%04d (%06f)"%\
			(epoch, epochs, loss_disc_2, correct_2, 2*len(dataset["train"]), correct_2*50.0/float(len(dataset["train"]))))
		print("[Clsfr] Epoch %04d/%04d : Loss : %06f \t Accuracy : %04d/%04d (%06f)"%\
			(epoch, epochs, loss_clsfr, correct_3, len(dataset["train"]), correct_3*100.0/float(len(dataset["train"]))))
		print("[Dec] Epoch %04d/%04d : Loss : %06f"%\
			(epoch, epochs, loss_rec))
		eval_model(enc, clsfr, dataset, device, "eval")
		print()

	# -------------------------------------------------- #

	# VISUALIZE
	visualize_embedding(enc, dataset, device, "eval", "mdl_13", weighted, False, w_n, thresh, beta)

	# -------------------------------------------------- #

	return enc, dec, disc, clsfr


def main():

	device = get_device(True)
	dataset = getDummyDataset()

	conv = [3,4,8,16]
	fc = [32,16,2]
	shape = dataset["train"].shape

	enc = EncoderNetwork(conv, shape).to(device)
	dec = DecoderNetwork(conv[::-1], enc.out_p).to(device)
	disc = FullyConnectedNetwork(fc, enc.size+dataset["train"].num_classes).to(device) # to append classes
	clsfr = FullyConnectedNetwork(fc, enc.size).to(device)

	enc, dec, disc, clsfr = train(enc, dec, disc, clsfr, dataset, device)

	# save_model(enc, "enc13.pt")
	# save_model(dec, "dec13.pt")
	# save_model(disc, "disc13.pt")
	# save_model(clsfr, "clsfr13.pt")

main()
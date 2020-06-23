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
	lamda = 1.0
	variance = 0.1

	weighted = True # weighted updates for clsfr
	binary = True # [0-1 weighting]
	thresh = 0.005 # threshold on loss value [for binary weighting]
	beta = 1e-4 # for multiplication in gaussian num
	w_n = True # weight normalization


	# if not using pretrained
	if dec is not None:

		# Loss Functions
		DiscLoss = nn.CrossEntropyLoss().to(device)
		RecreationLoss = nn.MSELoss().to(device)

		# Optimizers
		disc_optim = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))
		main_optim = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=0.0002, betas=(0.5, 0.999))

		# iterate for epochs [TRAINING PART 1]
		for epoch in range(1, epochs+1):

			# set the flags to train
			enc.train()
			dec.train()
			disc.train()

			# get the data loader
			dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)

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
				disc_optim.zero_grad()
				enc.zero_grad()
				disc.zero_grad()

				# get hidden and reshape
				hidden = enc(data).view(data.shape[0], -1)

				# sample prior according to target
				prior = (torch.randn(hidden.shape)*variance).to(device)
				prior = prior + 2*target.repeat(prior.shape[1], 1).T - 1 # just add the class index to mean

				# concatenate to get X and Y
				X = torch.cat([hidden, prior])
				Y = torch.cat([torch.zeros(hidden.shape[0]), torch.ones(hidden.shape[0])]).long().to(device)

				# update X according to the target
				# append the one-hot vector of target

				# calculate the one-hot vector
				idx = torch.Tensor([[i,target[i]] for i in range(len(target))]).long()
				OH = torch.zeros((len(target), dataset["train"].num_classes))
				OH[idx[:,0], idx[:,1]] = 1
				OH = OH.to(device)
				OH = torch.cat([OH,OH])

				# append to X
				X = torch.cat([X,OH], 1)

				# get output of discriminator
				out = disc(X)

				# calculate loss and update params
				loss1 = DiscLoss(out, Y)
				loss1.backward()
				disc_optim.step()

				# get accuracy and loss_disc_1
				correct = torch.sum(Y == torch.argmax(out, 1))
				loss_disc_1 += len(X)*loss1
				correct_1 += correct

				# TRAIN ENCODER AND DECODER
				# 0 means it is from encoder
				# 1 means it is from prior

				# set gradients to zero
				main_optim.zero_grad()
				enc.zero_grad()
				dec.zero_grad()
				disc.zero_grad()

				# get hidden
				hidden = enc(data)

				# add reconstruction error to loss
				data_ = dec(hidden)
				loss2 = RecreationLoss(data, data_)
				loss_rec += len(data)*loss2

				# reshape hidden and get prior according to target
				hidden = hidden.view(data.shape[0], -1)

				# sample prior according to target
				prior = (torch.randn(hidden.shape)*variance).to(device)
				prior = prior + 2*target.repeat(prior.shape[1], 1).T - 1 # just add the class index to mean

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
				X = torch.cat([X,OH], 1).to(device)

				# get output of discriminator and update it
				out = disc(X)

				# calculate loss and update params
				loss3 = DiscLoss(out, Y)
				loss_disc_2 += len(X)*loss3

				loss = loss2 - lamda*loss3
				loss.backward()
				main_optim.step()

				# get accuracy and loss_disc_2
				correct = torch.sum(Y == torch.argmax(out, 1))
				correct_2 += correct

				print(".", end="")


			loss_disc_1 = loss_disc_1/(2*len(dataset["train"]))
			loss_disc_2 = loss_disc_2/(2*len(dataset["train"]))
			loss_rec = loss_rec/len(dataset["train"])
			
			# Pretty Printing
			print("")
			print("[Disc1] Epoch %04d/%04d : Loss : %06f \t Accuracy : %04d/%04d (%06f)"%\
				(epoch, epochs, loss_disc_1, correct_1, 2*len(dataset["train"]), correct_1*50.0/float(len(dataset["train"]))))
			print("[Disc2] Epoch %04d/%04d : Loss : %06f \t Accuracy : %04d/%04d (%06f)"%\
				(epoch, epochs, loss_disc_2, correct_2, 2*len(dataset["train"]), correct_2*50.0/float(len(dataset["train"]))))
			print("[Dec] Epoch %04d/%04d : Loss : %06f"%\
				(epoch, epochs, loss_rec))
			print()

			# eval_model(enc, disc, dataset, device)
			# print()

	# -------------------------------------------------- #

	# VISUALIZE
	visualize_embedding(enc, dataset, device, "eval", "mdl_10", weighted, binary, w_n, thresh, beta)

	# -------------------------------------------------- #

	# Loss Function
	ClsfrLoss = None # placeholder
	if weighted is True:
		ClsfrLoss = nn.CrossEntropyLoss(reduction='none').to(device)
	else:
		ClsfrLoss = nn.CrossEntropyLoss(reduction='mean').to(device)

	# Optimizer
	clsfr_optim = optim.Adam(clsfr.parameters(), lr=0.0002, betas=(0.5, 0.999))

	# iterate for epochs [TRAINING PART 2]
	for epoch in range(1, epochs+1):

		# set the flags
		enc.eval()
		clsfr.train()

		# get the data loader
		dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)

		# initialize loss values
		loss_clsfr = 0.
		correct_3 = 0
		num_pts = 0

		# iterate over mini batches
		for data, target in dataloader:

			# put data and target to device
			data = data.to(device)
			target = target.to(device)

			# data.shape = [n,3,l,b]
			# target.shape = [n]

			# set gradients to zero
			clsfr_optim.zero_grad()
			enc.zero_grad()
			clsfr.zero_grad()

			# get hidden
			hidden = enc(data).view(data.shape[0], -1)

			# getting weights
			weights = None # placeholder
			if weighted is True:

				weights = hidden - target.repeat(hidden.shape[1], 1).T
				weights = beta*weights*weights
				weights = torch.sum(weights, 1)
				weights = torch.exp(-1*weights)
				# print("Before Normalization", weights)

				if w_n is True:
					weights = weights / torch.sum(weights)

				# print("After Normalization", weights)

				if binary is True:
					weights = (weights > thresh).long()
					num_pts += torch.sum(weights)

				# print("After Binary", weights)

			# get output of classifier
			out = clsfr(hidden)

			# calculate loss
			loss4 = ClsfrLoss(out, target)

			# multiply by weights (if reqd)
			if weighted is True:
				loss4 = torch.mean(loss4*weights)

			# update params
			loss4.backward()
			clsfr_optim.step()

			# get accuracy and loss_epoch
			correct = torch.sum(target == torch.argmax(out, 1))
			loss_clsfr += len(data)*loss4
			correct_3 += correct

			print(".", end="")

		loss_clsfr = loss_clsfr/(len(dataset["train"]))
		
		# Pretty Printing
		print("")
		if binary and weighted is True:
			print("Using %04d/%04d (%06f) points"%(num_pts, len(dataset["train"]), num_pts*100.0/float(len(dataset["train"]))))
		print("[Clsfr] Epoch %04d/%04d : Loss : %06f \t Accuracy : %04d/%04d (%06f)"%\
			(epoch, epochs, loss_clsfr, correct_3, len(dataset["train"]), correct_3*100.0/float(len(dataset["train"]))))
		eval_model(enc, clsfr, dataset, device, "eval")		
		print()

	return enc, dec, disc, clsfr


def main():

	device = get_device(True)
	dataset = getBreakhisDataset()

	conv = [3,4,8,16]
	fc = [32,16,2]
	shape = dataset["train"].shape

	# enc = EncoderNetwork(conv, shape).to(device)
	enc = torch.load("models/enc5.pt").to(device)

	# dec = DecoderNetwork(conv[::-1], enc.out_p).to(device)
	# disc = FullyConnectedNetwork(fc, enc.size+dataset["train"].num_classes).to(device) # to append classes

	clsfr = FullyConnectedNetwork(fc, enc.size).to(device)

	# enc, dec, disc, clsfr = train(enc, dec, disc, clsfr, dataset, device)
	enc, dec, disc, clsfr = train(enc, None, None, clsfr, dataset, device)

	save_model(enc, "enc11.pt")
	save_model(dec, "dec11.pt")
	save_model(disc, "disc11.pt")
	save_model(clsfr, "clsfr11.pt")

main()
import torch
import torch.nn as nn
import os
import random
import matplotlib.pyplot as plt

def get_device(cuda):
	"""
	returns the device used in the system (cpu/cuda)
	"""
	device = torch.device("cuda" if torch.cuda.is_available() and cuda == 1 else "cpu")
	print("Using Device : " + str(device))
	return device

def eval_model(enc, disc, dataset, device, folder):
	"""
	Used to evaluate a model after training.
	"""

	batch_size = 50

	# Loss Functions
	DiscLoss = nn.CrossEntropyLoss().to(device)

	enc.eval()
	disc.eval()

	# get the data loader
	dataloader = torch.utils.data.DataLoader(dataset["eval"], batch_size=batch_size, shuffle=True)

	loss_epoch = 0.
	correct_epoch = 0

	for data, target in dataloader:

		# data.shape = [n,3,l,b]
		# target.shape = [n]

		# put datasets on the device
		data = data.to(device)
		target = target.to(device)

		# get output of discriminator
		hidden = enc(data).view(data.shape[0], -1)
		out = disc(hidden)

		# calculate loss and update params
		loss = DiscLoss(out, target)

		# get accuracy and loss_epoch
		correct = torch.sum(target == torch.argmax(out, 1))

		loss_epoch += len(data)*loss
		correct_epoch += correct

	loss_epoch = loss_epoch/len(dataset[folder])
	
	# Pretty Printing
	print("Loss : %06f \t Accuracy : %04d/%04d (%06f)"%\
		(loss_epoch, correct_epoch, len(dataset["eval"]), correct_epoch*100.0/float(len(dataset[folder]))))
	
def save_model(model, name):
	if "models" not in os.listdir():
		os.mkdir("models")
	torch.save(model, "models/"+name)

def visualize_embedding(enc, dataset, device, folder):
	"""
	To visualize the embeddings of the encoder
	"""

	variance = 1
	batch_size = 50
	num_graphs = 5
	enc.eval()

	# get the data loader
	dataloader = torch.utils.data.DataLoader(dataset[folder], batch_size=batch_size, shuffle=False)

	H = []
	T = []

	for data, target in dataloader:

		# data.shape = [n,3,l,b]
		# target.shape = [n]

		# put datasets on the device
		data = data.to(device)
		target = target.to(device)

		# get output of discriminator
		hidden = enc(data).view(data.shape[0], -1)

		H.append(hidden)
		T.append(target)

	H = torch.cat(H,0)
	T = torch.cat(T,0)

	prior = (torch.randn(H.shape)*variance).to(device)
	# prior = prior + T.repeat(prior.shape[1], 1).T # just add the class index to mean

	H = H.detach().cpu().numpy()
	T = T.detach().cpu().numpy()
	prior = prior.detach().cpu().numpy()

	for idx in range(num_graphs):

		idxs = random.sample(range(H.shape[1]), k=2)
		print("Using Indices "+str(idxs))

		# The actual plotting
		# RED : hidden 0
		# GREEN : prior 0, 1
		# BLUE : hidden 1

		plt.plot(prior[:,idxs[0]], prior[:,idxs[1]], 'go')
		for i in range(len(H)):
			if T[i] == 0:
				plt.plot(H[i,idxs[0]], H[i,idxs[1]], 'ro')
			else:
				plt.plot(H[i,idxs[0]], H[i,idxs[1]], 'bo')

		plt.savefig("plots/1_%d.jpg"%(idx+1))
		plt.show()

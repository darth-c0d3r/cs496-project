import torch
import torch.nn as nn

def get_device(cuda):
	"""
	returns the device used in the system (cpu/cuda)
	"""
	device = torch.device("cuda" if torch.cuda.is_available() and cuda == 1 else "cpu")
	print("Using Device : " + str(device))
	return device

def BootstrapLoss(out, target, weights, device):
	"""
	Calculates and returns the Bootstrap Loss
	out : [n x 2]
	target : [n]
	weights : [n]
	"""

	idx = torch.Tensor([[i,target[i]] for i in range(len(target))]).long()
	Y = torch.zeros((len(target), len(out[0]))).to(device)
	Y[idx[:,0], idx[:,1]] = 1

	pred = torch.argmax(out, 1)

	idx = torch.Tensor([[i,pred[i]] for i in range(len(pred))]).long()
	Z = torch.zeros((len(pred), len(out[0]))).to(device)
	Z[idx[:,0], idx[:,1]] = 1

	h = torch.softmax(out, 1)
	W = weights.repeat(2,1).T

	loss = -1*torch.mean((W*Y + (1-W)*Z)*torch.log(h))

	return loss
		
			


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
	dataloader = torch.utils.data.DataLoader(dataset[folder], batch_size=batch_size, shuffle=True)

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
	print("[%s] Loss : %06f \t Accuracy : %04d/%04d (%06f)"%\
		(folder, loss_epoch, correct_epoch, len(dataset[folder]), correct_epoch*100.0/float(len(dataset[folder]))))
	
def save_model(model, name):
	if "models" not in os.listdir():
		os.mkdir("models")

	if model is not None:
		torch.save(model, "models/"+name)

def visualize_embedding(enc, dataset, device, folder, directory, weighted, binary, w_n, thresh, beta):
	"""
	To visualize the embeddings of the encoder
	"""

	variance = 0.1
	batch_size = 50
	num_graphs = 20
	enc.eval()

	# get the data loader
	dataloader = torch.utils.data.DataLoader(dataset[folder], batch_size=batch_size, shuffle=False)

	H = []
	T = []
	W = []

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

		# getting weights
		weights = None # placeholder
		if weighted is True and binary is True:

			weights = hidden - (2*target.repeat(hidden.shape[1], 1).T-1)
			weights = beta*weights*weights
			weights = torch.sum(weights, 1)
			weights = torch.exp(-1*weights)

			if w_n is True:
				weights = weights / torch.sum(weights)

			weights = (weights > thresh).long()

			W.append(weights)


	H = torch.cat(H,0)
	T = torch.cat(T,0)

	if weighted is True and binary is True:
		W = torch.cat(W,0)

	prior = (torch.randn(H.shape)*variance).to(device)
	prior = prior + 2*T.repeat(prior.shape[1], 1).T - 1 # just add the class index to mean

	H = H.detach().cpu().numpy()
	T = T.detach().cpu().numpy()
	prior = prior.detach().cpu().numpy()

	if "plots" not in os.listdir():
		os.mkdir("plots")

	if directory not in os.listdir("plots"):
		os.mkdir("plots/"+directory)

	for idx in range(num_graphs):

		idxs = random.sample(range(H.shape[1]), k=2)
		print("Using Indices "+str(idxs))

		# The actual plotting
		# RED : hidden 0
		# GREEN : prior 0
		# BLUE : hidden 1
		# YELLOW : prior 1

		if weighted is False or binary is False:
			for i in range(len(H)):
				if T[i] == 0:
					plt.plot(H[i,idxs[0]], H[i,idxs[1]], 'ro', markersize=2)
					plt.plot(prior[i,idxs[0]], prior[i,idxs[1]], 'go', markersize=2)
				else:
					plt.plot(H[i,idxs[0]], H[i,idxs[1]], 'bo', markersize=2)
					plt.plot(prior[i,idxs[0]], prior[i,idxs[1]], 'yo', markersize=2)
		else:
			for i in range(len(H)):
				if T[i] == 0:
					if W[i] == 1:
						plt.plot(H[i,idxs[0]], H[i,idxs[1]], 'ro', markersize=2)
					else:
						plt.plot(H[i,idxs[0]], H[i,idxs[1]], 'r+', markersize=2)
					plt.plot(prior[i,idxs[0]], prior[i,idxs[1]], 'go', markersize=2)
				else:
					if W[i] == 1:
						plt.plot(H[i,idxs[0]], H[i,idxs[1]], 'bo', markersize=2)
					else:
						plt.plot(H[i,idxs[0]], H[i,idxs[1]], 'b+', markersize=2)
					plt.plot(prior[i,idxs[0]], prior[i,idxs[1]], 'yo', markersize=2)

		plt.savefig("plots/%s/%d_(%d_%d).jpg"%(directory,idx+1,idxs[0], idxs[1]))
		plt.show()

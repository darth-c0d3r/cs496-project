import torch
import torch.nn as nn

def get_device(cuda):
	"""
	returns the device used in the system (cpu/cuda)
	"""
	device = torch.device("cuda" if torch.cuda.is_available() and cuda == 1 else "cpu")
	print("Using Device : " + str(device))
	return device

def eval_model(enc, disc, dataset, device):
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

		# get output of discriminator
		hidden = enc(data).view(data.shape[0], -1)
		out = disc(hidden)

		# calculate loss and update params
		loss = DiscLoss(out, target)

		# get accuracy and loss_epoch
		correct = torch.sum(target == torch.argmax(out, 1))

		loss_epoch += len(data)*loss
		correct_epoch += correct

	loss_epoch = loss_epoch/len(dataset["train"])
	
	# Pretty Printing
	print("Loss : %06f \t Accuracy : %04d/%04d (%06f)"%\
		(loss_epoch, correct_epoch, len(dataset["eval"]), correct_epoch*100.0/float(len(dataset["eval"]))))
	
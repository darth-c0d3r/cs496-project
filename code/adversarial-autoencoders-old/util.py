import torch

def get_device(cuda):
	"""
	returns the device used in the system (cpu/cuda)
	"""
	device = torch.device("cuda" if torch.cuda.is_available() and cuda == 1 else "cpu")
	print("Using Device : " + str(device))
	return device

def get_onehot(target):
	"""
	returns one hot vector corresponding to target which is n x 1 tensor
	"""
	onehot = torch.zeros((len(target),2))
	for idx in range(len(target)):
		onehot[idx][target[idx][0]] = 1

	return onehot
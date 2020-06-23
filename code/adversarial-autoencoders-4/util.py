import torch
import torch.nn as nn

def get_device(cuda):
	"""
	returns the device used in the system (cpu/cuda)
	"""
	device = torch.device("cuda" if torch.cuda.is_available() and cuda == 1 else "cpu")
	print("Using Device : " + str(device))
	return device

def eval_model(enc, dataset, device, num_classes, data_name):
	"""
	Used to evaluate a model after training.
	"""
	
	batch_size = 50

	grid = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

	for cl in range(num_classes):

		# get the data loader
		dataloader = torch.utils.data.DataLoader(dataset[cl][data_name], batch_size=batch_size, shuffle=True)

		for data, target in dataloader:

			distances = [None]*num_classes

			for cl in range(num_classes):

				enc[cl].eval()

				hidden = enc[cl](data).view(data.shape[0], -1)
				distances[cl] = torch.mean(hidden*hidden, 1).view(1,-1)

			distances = torch.cat(distances)
			pred = torch.argmax(distances, 0)

			for p in pred:
				grid[cl][p] += 1

	print(grid)

	correct = sum([grid[cl][cl] for cl in range(num_classes)])
	total = sum([sum(grid[cl]) for cl in range(num_classes)])

	print("%04d/%04d (%06f)"%(correct, total, 100.0*correct/total))

import torch
import torch.nn as nn

def evaluate_conv(enc, clsfr, dataset, device, epoch=0):


	batch_size = 50
	ClsfrLoss = nn.CrossEntropyLoss().to(device)
	dataloader = torch.utils.data.DataLoader(dataset["eval"], batch_size=batch_size, shuffle=True)

	enc.eval()
	clsfr.eval()

	total = 0
	num_steps = 0
	epoch_loss = 0.
	epoch_correct = 0

	for data, target in dataloader:

		data = data.to(device)
		target = target.to(device)

		# data.shape = [n,3,l,b]
		# target.shape = [n,1]
		target = target.squeeze(-1)

		out = clsfr(enc(data))
		loss = ClsfrLoss(out, target)
		labels = torch.argmax(out, 1)
		correct = torch.sum(labels == target)

		epoch_loss = (epoch_loss*num_steps + loss)/(num_steps + 1)
		total += len(data)
		num_steps += 1
		epoch_correct += correct

	print("Epoch [Eval]: %004d \t Loss: %004f \t Correct: %004d/%004d (%004f)"%\
			(epoch, epoch_loss, epoch_correct, total, float(epoch_correct)/float(total)))
	
import torch
import torch.nn as nn

class EncoderNetwork(nn.Module):
	"""
	Encodes an input batch of images to smaller dimensions.
	"""

	def __init__(self, conv, input_size):
		"""
		conv is the list of number of kernels in each layer.
		eg. conv = [3,4,8,16,32,32]
		IMPORTANT: Include Input Channels in the conv list

		input_size is a 3-tuple containing the size of the input.
		eg. input_size = [3,28,28]
		"""

		super(EncoderNetwork, self).__init__()

		# model parameters
		kernel_size = 5
		stride = 2
		padding = 1
		dropout_rate = 0.3
		leaky_relu_slope = 0.2
		self.size = input_size

		# list required to make sure that decoder has same output dimensions
		output_padding = []

		# Module Lists to contain the layers
		self.batchnorm_layers = nn.ModuleList()
		self.dropout_layers = nn.ModuleList()
		self.conv_layers = nn.ModuleList()
		self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)

		for i in range(len(conv)-1):

			self.conv_layers.append(nn.Conv2d(conv[i], conv[i+1], kernel_size, stride, padding))
			self.batchnorm_layers.append(nn.BatchNorm2d(conv[i+1]))
			self.dropout_layers.append(nn.Dropout2d(dropout_rate))

			# output size of the layer
			osize = [conv[i+1], ((self.size[1]+2*padding-kernel_size)//stride)+1, ((self.size[2]+2*padding-kernel_size)//stride)+1]

			output_padding += [(self.size[1]-((osize[1]-1)*stride - 2*padding + kernel_size), 
					self.size[2]-((osize[2]-1)*stride - 2*padding + kernel_size))]

			# print(self.size, self.size[0]*self.size[1]*self.size[2])
			self.size = osize

		print("Encoded Space Dimensions : " + str(self.size))

		self.output_padding = output_padding[::-1]
		self.size = self.size[0]*self.size[1]*self.size[2]

		print("Encoded Space Size : " + str(self.size))

	def forward(self, X):

		for idx in range(len(self.conv_layers)-1):

			conv = self.conv_layers[idx]
			batchnorm = self.batchnorm_layers[idx]
			dropout = self.dropout_layers[idx]

			# comment out what is not needed
			X = conv(X)
			X = batchnorm(X)
			X = self.leaky_relu(X)
			X = dropout(X)

		X = self.conv_layers[-1](X)

		return X


class DecoderNetwork(nn.Module):

	def __init__(self, deconv, output_padding):
		"""
		deconv is the reverse of the conv list used in EncoderNetwork
		output_padding is obtained from EncoderNetwork
		"""

		super(DecoderNetwork, self).__init__()

		# model parameters
		kernel_size = 5 
		stride = 2
		padding = 1
		# the above 3 should be same as the corresponding EncoderNetwork
		dropout_rate = 0.3
		leaky_relu_slope = 0.2

		self.batchnorm_layers = nn.ModuleList()
		self.deconv_layers = nn.ModuleList()
		self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)

		for i in range(len(deconv)-1):
			self.deconv_layers.append(nn.ConvTranspose2d(deconv[i], deconv[i+1], kernel_size, stride, padding, output_padding[i]))
			self.batchnorm_layers.append(nn.BatchNorm2d(deconv[i+1]))

	def forward(self, X):

		for deconv, batchnorm in zip(self.deconv_layers[:-1], self.batchnorm_layers[:-1]):
			X = self.leaky_relu(batchnorm(deconv(X)))
		X = torch.tanh(self.deconv_layers[-1](X))

		return X


class FullyConnectedNetwork(nn.Module):

	def __init__(self, fc, size_out):
		"""
		fc is the number of neurons per FC Layer
		Important: input size has to be included in the list fc
		
		size_out is the number of output neurons (often the number of classes)

		"""

		super(FullyConnectedNetwork, self).__init__()

		leaky_relu_slope = 0.2
		dropout_rate = 0.4

		self.batchnorm_layers = nn.ModuleList()
		self.dropout_layers = nn.ModuleList()
		self.fc_layers = nn.ModuleList()
		self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)

		for i in range(len(fc)-1):
			self.fc_layers.append(nn.Linear(fc[i], fc[i+1]))
			self.batchnorm_layers.append(nn.BatchNorm1d(fc[i+1]))
			self.dropout_layers.append(nn.Dropout(dropout_rate))

		self.output_layer = nn.Linear(fc[-1], size_out)

	def forward(self, X):
		X = X.view(X.shape[0], -1)

		for idx in range(len(self.fc_layers)):

			fc_layer = self.fc_layers[idx]
			batchnorm = self.batchnorm_layers[idx]
			dropout = self.dropout_layers[idx]

			X = fc_layer(X)
			X = batchnorm(X)
			X = self.leaky_relu(X)
			X = dropout(X)
			
		X = self.output_layer(X)

		return X



# For some dummy testing

data = torch.randn((5,3,64,64))
conv = [3, 4, 8]
# fc = [32,16,8,2]

enc = EncoderNetwork(conv, list(data[0].shape))
dec = DecoderNetwork(conv[::-1], enc.out_p)
# clsr = FullyConnectedNetwork(fc, enc.size)

print(data.shape)
h = enc(data)
print(h.shape)
out = dec(h)
print(out.shape)

# h = h.view(h.shape[0], -1)

# out = clsr(h)
# print(out.shape)
# print(out)
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetCatVsDogs(nn.Module):
	def __init__(self):
		super().__init__()
		self.convolution1 = nn.Conv2d(1, 32, 5)
		self.convolution2 = nn.Conv2d(32, 64, 5)
		self.convolution3 = nn.Conv2d(64, 128, 5)
		
		x = torch.randn(50, 50).view(-1, 1, 50, 50)
		self._to_linear = None
		self.convolutions(x)
		
		self.fc1 = nn.Linear(self._to_linear, 512)
		self.fc2 = nn.Linear(512, 2)
	
	def convolutions(self, x):
		x = F.max_pool2d(F.relu(self.convolution1(x)), (2, 2))
		x = F.max_pool2d(F.relu(self.convolution2(x)), (2, 2))
		x = F.max_pool2d(F.relu(self.convolution3(x)), (2, 2))
		
		print(x[0].shape)
		
		if self._to_linear is None:
			self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
		return x
	
	def forward(self, x):
		x = self.convolutions(x)
		x = x.view(-1, self._to_linear)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.softmax(x, dim=1)




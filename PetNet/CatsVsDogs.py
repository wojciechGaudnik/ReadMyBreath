import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from torch import nn
from tqdm import tqdm

from PetNet.NetCatVsDogs import NetCatVsDogs


class DogsVSCats():
	IMG_SIZE = 50
	CATS = "PetImages/Cat"
	DOGS = "PetImages/Dog"
	LABELS = {CATS: 0, DOGS: 1}
	training_data = []
	_cat_count = 0
	_dog_count = 0
	BATCH_SIZE = 100
	EPOCHS = 2
	
	def __init__(self, device="cpu"):
		self.depth = 1
		self.path_main = os.getcwd()
		self.print_u("Init")
		self.device = torch.device(device)
		self.print_u("Init Net")
		self.net_cat_vs_dogs = NetCatVsDogs().to(device)
		self.optimizer = optim.Adam(self.net_cat_vs_dogs.parameters(), lr=0.001)
		self.loss_function = nn.MSELoss()
		self.print_u("Init Net DONE")
		self.print_u("Init DONE")
	
	def make_training_data(self):
		self.print_u("Make data")
		for label in self.LABELS:
			print(label)
			for f in tqdm(os.listdir(label)):
				try:
					path = os.path.join(label, f)
					img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
					img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
					self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])
					if label == self.CATS:
						self._cat_count += 1
					elif label == self.DOGS:
						self._dog_count += 1
				except Exception:
					pass
		
		np.random.shuffle(self.training_data)
		np.save(f'{self.path_main}/PetTrainingData/training_data.npy', self.training_data)
		print(f'Cats:\t{self._cat_count}')
		print(f'Dogs:\t{self._dog_count}')
		self.print_u("Make data DONE")
	
	def split_training_data(self):
		self.print_u("Split training data")
		training_data = np.load(f'{self.path_main}/PetTrainingData/training_data.npy', allow_pickle=True)
		plt.imshow(training_data[1][0], cmap="gray")
		plt.show()
		X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
		X = X / 255.0
		y = torch.Tensor([i[1] for i in training_data])
		VAL_PCT = 0.1
		val_size = int(len(X) * VAL_PCT)
		train_X = X[:-val_size]
		train_y = y[:-val_size]
		test_X = X[-val_size:]
		test_y = y[-val_size:]
		print(f'Size of all data\t{val_size}')
		print(f'Size of train data:\t{len(train_X)}')
		print(f'Size of test data:\t{len(test_X)}')
		self.print_u("Split training data DONE")
		return test_X, test_y, train_X, train_y
	
	def train(self):
		self.print_u(f'Train on {"GPU" if torch.cuda.is_available() else "CPU"}')
		self.test_X, self.test_y, self.train_X, self.train_y = self.split_training_data()
		for epoch in range(self.EPOCHS):
			print(f'Epoch: {epoch}        ')
			for i in tqdm(range(0, len(self.train_X), self.BATCH_SIZE)):
				batch_X = self.train_X[i: i + self.BATCH_SIZE].view(-1, 1, 50, 50)
				batch_y = self.train_y[i: i + self.BATCH_SIZE]
				batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
				
				self.optimizer.zero_grad()
				outputs = self.net_cat_vs_dogs(batch_X)
				loss = self.loss_function(outputs, batch_y)
				loss.backward()
				self.optimizer.step()
		self.print_u("Train DONE")
	
	def test(self):
		correct = 0
		total = 0
		with torch.no_grad():
			for i in tqdm(range(len(self.test_X))):
				real_class = torch.argmax(self.test_y[i]).to(self.device)
				net_out = self.net_cat_vs_dogs(self.test_X[i].view(-1, 1, 50, 50).to(self.device))[0]
				predicted_class = torch.argmax(net_out)
				if predicted_class == real_class:
					correct += 1
				total += 1
		print("Accuracy:", round(correct / total, 3))
	
	def fwd_pass(self, X, y, train=False):
		if train:
			self.net_cat_vs_dogs.zero_grad()
		outputs = self.net_cat_vs_dogs(X)
		matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
		accuracy = matches.count(True) / len(matches)
		loss = self.loss_function(outputs, y)
		
		if train:
			loss.backward()
			self.optimizer.step()
		return accuracy, loss
	
	def print_u(self, text):
		self.depth += -2 if 'DONE' in text else 2
		print(f'{"-" * 1 + "-"  * self.depth + text + "-" * (60 - len(text) - self.depth)}')
		self.depth += -2 if 'DONE' in text else 2





import os
import sys

import cv2
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim

from NetCatVsDogs import NetCatVsDogs

REBUILD_DATA = False

class DogsVSCats():
	IMG_SIZE = 50
	CATS = "PetImages/Cat"
	DOGS = "PetImages/Dog"
	LABELS = {CATS: 0, DOGS: 1}
	training_data = []
	_cat_count = 0
	_dog_count = 0
	
	def make_training_data(self):
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
				except Exception as e:
					print(str(e))
		
		np.random.shuffle(self.training_data)
		np.save("training_data.npy", self.training_data)
		
	def test(self, device):
		print("Cats:", self._cat_count)
		print("Dogs:", self._dog_count)
		training_data = np.load("training_data.npy", allow_pickle=True)
		print("doneee1")
		# print(len(training_data[1]))
		print((training_data[0][0]))
		plt.imshow(training_data[5][0], cmap="gray")
		plt.show()
		print("doneee2")
		# plt.imshow()
		net_cat_vs_dogs = NetCatVsDogs().to(device)
		# net_cat_vs_dogs.to(device)

		X = torch.Tensor([i[0] for i in training_data]).view(-1, 50,50)
		X = X/255.0
		y = torch.Tensor([i[1] for i in training_data])
		VAL_PCT = 0.1
		val_size = int(len(X) * VAL_PCT)
		print(val_size)
		
		train_X = X[:-val_size]
		train_y = y[:-val_size]

		test_X = X[-val_size:]
		test_y = y[-val_size:]
		
		print(len(train_X))
		print(len(test_X))
		
		BATCH_SIZE = 100
		EPOCHS = 10
		optimizer = optim.Adam(net_cat_vs_dogs.parameters(), lr=0.001)
		loss_function = nn.MSELoss()
		for epoch in range(EPOCHS):
			for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
				print("Epoch: ", i)
				batch_X = train_X[i: i + BATCH_SIZE].view(-1, 1, 50, 50)
				batch_y = train_y[i: i + BATCH_SIZE]
				batch_X, batch_y = batch_X.to(device), batch_y.to(device)

				
				optimizer.zero_grad()
				outputs = net_cat_vs_dogs(batch_X)
				loss = loss_function(outputs, batch_y)
				loss.backward()
				optimizer.step()
		print(loss)
		print("-------")
		
		correct = 0
		total = 0
		with torch.no_grad():
			for i in tqdm(range(len(test_X))):
				real_class = torch.argmax(test_y[i]).to(device)
				net_out = net_cat_vs_dogs(test_X[i].view(-1, 1, 50, 50).to(device))[0]
				predicted_class = torch.argmax(net_out)
				if predicted_class == real_class:
					correct += 1
				total += 1
		print("Accuracy:", round(correct/total,3))
		
			

dogVsCats = DogsVSCats()
if REBUILD_DATA:
	device = torch.device("cuda:0")
	# device = torch.device("cpu")
	dogVsCats.make_training_data()
else:
	print(torch.cuda.is_available())
	print(torch.cuda.device_count())
	device = torch.device("cuda:0")
	# device = torch.device("cpu")
	dogVsCats.test(device)
	print("start")
	# print(device.index)
	



# tqdm.clear()
# tqdm.close()

print("end")
sys.exit()




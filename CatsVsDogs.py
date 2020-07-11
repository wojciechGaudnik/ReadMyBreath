import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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
		
	def test(self):
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


dogVsCats = DogsVSCats()
if REBUILD_DATA:
	dogVsCats.make_training_data()
else:
	dogVsCats.test()

		
		

# tqdm.clear()
# tqdm.close()




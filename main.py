import sys

from PetNet.CatsVsDogs import DogsVSCats
from PetNet.DataGraph import DataGraph


train = True
graph = True

def main():
	if train:
		dogVsCats = DogsVSCats("cuda:0")
		# dogVsCats.make_training_data()
		# dogVsCats.train_simple()
		# dogVsCats.test()
		dogVsCats.split_training_data()
		dogVsCats.train()
		# val_acc, val_lost = dogVsCats.test_accuracy_lost()
		# print(val_acc, val_lost)
	if graph:
		test = DataGraph(dogVsCats.MODEL_NAME)
		test.create_accuracy_loss_graph()
	sys.exit()



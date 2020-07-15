import os

import matplotlib.pyplot as plt
from matplotlib import style



class DataGraph:
	def __init__(self, model_name):
		style.use("ggplot")
		self.model_name = model_name
		self.path_main = os.getcwd()
		
	def create_accuracy_loss_graph(self):
		contents = open(f'{self.path_main}/model.log', "r").read().split('\n')
		
		times = []
		accuracies = []
		losses = []
		
		values_accuracies = []
		values_losses = []
		
		for c in contents:
			if self.model_name in c:
				name, time_stamp, accuracy, loss, values_accuracy, values_loss = c.split(",")
				times.append(float(time_stamp))
				accuracies.append(float(accuracy))
				losses.append(float(loss))
				values_accuracies.append(float(values_accuracy))
				values_losses.append(float(values_loss))
				
		fig = plt.figure()
		ax1 = plt.subplot2grid((2,1), (0,0))
		ax2 = plt.subplot2grid((2,1), (1, 0), sharex=ax1)
		
		ax1.plot(times, accuracies, label="accuracy")
		ax1.plot(times, values_accuracies, label="value_accuracy")
		ax1.legend(loc=2)
		
		ax2.plot(times, losses, label="loss")
		ax2.plot(times, values_losses, label="value_loss")
		ax2.legend(loc=2)
		
		plt.show()

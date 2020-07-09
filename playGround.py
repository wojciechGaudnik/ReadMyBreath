import torch
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


from torchvision import transforms, datasets

from Net import Net

x = torch.Tensor([5, 3])
y = torch.Tensor([2, 1])

print(x * y)

x = torch.zeros([2,5])
print(x)

print(x.shape)

y = torch.rand([2,5])
print(y)

y = y.view([1,10])
print(y)

train = datasets.MNIST("", train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True, transform = transforms.Compose([transforms.ToTensor()]))

trainSet = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testSet = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

print(trainSet)
for data in trainSet:
	print(data)
	break

x, y = data[0][0], data[1][0]
print(y)

print(data[0][0].shape)
plt.imshow(data[0][0].view(28, 28))
plt.show()

total = 0
counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

# for data in trainSet:
# 	Xs, ys = data
# 	for y in ys:
# 		counter_dict[int(y)] += 1
# 		total += 1
#
# print(counter_dict)
#
# for i in counter_dict:
# 	print(f"{i}: {counter_dict[i]/total * 100}")
	

myNet = Net()
# print(myNet)
#
# X = torch.rand((28, 28))
# X = X.view(1, 28 * 28)
#
# output = myNet(X)

optimizer = optim.Adam(myNet.parameters(), lr=0.001)

EPOCHS = 2
for epoch in range(EPOCHS):
	for data in trainSet:
		X, y = data
		myNet.zero_grad()
		output = myNet(X.view(-1, 28*28))
		loss = F.nll_loss(output, y)
		loss.backward()
		optimizer.step()
	print(loss)
	
correct = 0
total = 0

with torch.no_grad():
	for data in trainSet:
		X, y = data
		output = myNet(X.view(-1, 28*28))
		for idx, i in enumerate(output):
			if torch.argmax(i) == y[idx]:
				correct += 1
			total += 1
			
print("Accuracy: ", round(correct / total, 3))
# plt.imshow(data[0][0].view(28, 28))
# plt.show()

plt.imshow(X[0].view(28, 28))
plt.show()
print(torch.argmax(myNet(X[0].view(-1, 28*28))[0]))



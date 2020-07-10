import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from Net import Net

train = datasets.MNIST("", train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True, transform = transforms.Compose([transforms.ToTensor()]))

trainSet = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testSet = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

myNet = Net()

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

plt.imshow(X[0].view(28, 28))
plt.show()
print(torch.argmax(myNet(X[0].view(-1, 28*28))[0]))



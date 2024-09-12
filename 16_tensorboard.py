# Based on 13_feed_forward_nn
from cProfile import label

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import sys

writer = SummaryWriter("runs/mnist2")

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size = 784  # 28x28 image size flattened
hidden_size = 500
num_classes = 10  # digits 0-9
num_epochs = 1
batch_size = 64
learning_rate = 0.01

# MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                           transform=transforms.ToTensor())  # ,download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                          transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = examples._next_data()

print(samples.shape, labels.shape)  # torch.Size([100, 1, 28, 28]) torch.Size([100]) 100 samples in a batch and 1 color

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(samples[i][0], cmap='gray')
# plt.show()
img_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnist_images', img_grid)
writer.close()
# sys.exit()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):  # num classes==output size
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # No softmax at the end
        return out


model = NeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
model = model.to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

writer.add_graph(model, samples.reshape(-1, 28*28).to(device))
writer.close()
# sys.exit()

# training loop
n_total_steps = len(train_loader)

running_loss = 0
running_corrects = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # [100, 1, 28, 28] --> [100, 784]
        images = images.reshape(-1, 28*28).to(device)  # reshape to 784 columns and as many rows as needed
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predictions = torch.max(outputs, 1)
        running_corrects += (predictions == labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f'epoch {epoch + 1} / {num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')
            writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i)
            writer.add_scalar('accuracy', running_corrects / 100, epoch * n_total_steps + i)
            running_loss = 0.0
            running_corrects = 0

labels = []
preds = []

# test and evaluation
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels1 in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)  # reshape to 784 columns and as many rows as needed
        labels1 = labels1.to(device)
        outputs = model(images)

        # .max return value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels1.shape[0]
        n_correct += (predictions == labels1).sum().item()

        class_prediction = [F.softmax(output, dim=0) for output in outputs]
        preds.append(class_prediction)
        labels.append(predictions)

    preds = torch.cat([torch.stack(batch) for batch in preds])
    labels = torch.cat(labels)

    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')

    classes = range(10)
    for i in classes:
        labels_i = labels == i
        preds_i = preds[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
        writer.close()

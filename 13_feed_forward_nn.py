# MNIST
# DataLoader, Transformation
# Multilayer Neural Network and activation functions
# Loss and optimizer
# Training Loop (batch training)
# Model evaluation
# GPU support
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size = 784  # 28x28 image size flattened
hidden_size = 100
num_classes = 10  # digits 0-9
num_epochs = 2
batch_size = 100
learning_rate = 0.001

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

# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(samples[i][0], cmap='gray')
# plt.show()

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

# training loop
n_total_steps = len(train_loader)
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

        if (i + 1) % 100 == 0:
            print(f'epoch {epoch + 1} / {num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')

# test and evaluation
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)  # reshape to 784 columns and as many rows as needed
        labels = labels.to(device)
        outputs = model(images)

        # .max return value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')

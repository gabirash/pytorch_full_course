"""
epoch = 1 forward and backward pass of ALL training samples
batch_size = number of training samples in one forward and backward pass
number of iterations = number of passes, each pass using [batch_size] number of samples

e.g. 100 samples with batch_size = 20 --> 100/20 = 5 iters. for 1 epoch.
"""
from pyexpat import features

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class WineDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt('./data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[: , [0]])  # n_samples but only the very first column (class or label)
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples

dataset = WineDataset()
# first_data = dataset[0]  # refers to the __getitem__ function from the class
# features, labels = first_data
# print(features, labels)
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)  # 2 multi-processes for faster load

# dataiter = iter(dataloader)
# data = dataiter._next_data()
# features, labels = data
# print(features, labels)  # We will see 4 features and 4 corresponding labels

# Dummy training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):  # refers to the __getitem__ function from the class
        # forward, backward and update
        if (i+1) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs{inputs.shape}')

# torchvision.datasets.MNIST()

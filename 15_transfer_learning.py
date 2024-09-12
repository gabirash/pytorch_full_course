"""
The first layers are characterized by Feature Learning, and the last ones (flatten --> ...) are for Classification.
Thus, we train only the last layers and by that create New Classification Layer.
In this example we will use the pre-trained resnet 18 cnn.
"""
from sched import scheduler

# ImagesFolder for datasets
# Scheduler to change the lr
# TL
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy


# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

# import data
data_dir = 'data/hymenoptera_data'
sets = ['train', 'val']
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in sets}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=4,
                             shuffle=True, num_workers=0)
               for x in sets}

dataset_sizes = {x: len(image_datasets[x]) for x in sets}
class_names = image_datasets['train'].classes
print(class_names)

# ... train model def etc...

# Fine-tune technique
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 2)  # 2 output classes (ants and bees)
model.to(device)

criterioin = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# scheduler
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # Every 7 epochs the lr will be mult. by 0.1
# for epoch in range(100):
#     train()
#     evaluate()
#     scheduler.step()

# model = train_model(model, criterion, ...)

# Freeze all layer in the beginning
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # Freeze all the layers in the beginning

num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 2)  # We create a new last layer (with grad = True by default)
model.to(device)

criterioin = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# scheduler
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # Every 7 epochs the lr will be mult. by 0.1

# model = train_model(model, criterion, ...)

import torch
import torch.nn as nn
import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
# print('softmax numpy:', outputs)

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)  # Computes it along the first output
# print('softmax torch:', outputs)


def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss  # / float(predicted.shape[0])


# y must be one-hot encoded
Y = np.array([1, 0, 0])

# y_pred has probabilities
y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, y_pred_good)
l2 = cross_entropy(Y, y_pred_bad)

# print(f'Loss1 numpy: {l1:.4f}')
# print(f'Loss2 numpy: {l2:.4f}')

loss = nn.CrossEntropyLoss()  # applies nn.LogSoftmax + nn.NLLLoss --> no Softmax in last layer!
# Also, Y has class labels, not One-Hot!
# Also, y_pred has raw scores (logits), no Softmax!

Y = torch.tensor([0])  # Class 0, instead of [1, 0, 0]
# size n_samples X n_classes = 1X3
y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])

l1 = loss(y_pred_good, Y)
l2 = loss(y_pred_bad, Y)

print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2.item():.4f}')

_, predictions1 = torch.max(y_pred_good, 1)
_, predictions2 = torch.max(y_pred_bad, 1)
# print(predictions1)  # Gives the class predicted - 0 (good)
# print(predictions2)  # Gives the class predicted - 1 (bad)

# 3 samples
Y = torch.tensor([2, 0, 1])
# size n_samples X n_classes = 3X3
y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])
y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1]])

l1 = loss(y_pred_good, Y)
l2 = loss(y_pred_bad, Y)

# print(f'Loss1 numpy: {l1:.4f}')
# print(f'Loss2 numpy: {l2.item():.4f}')

_, predictions1 = torch.max(y_pred_good, 1)
_, predictions2 = torch.max(y_pred_bad, 1)
# print(predictions1)  # Gives the class predicted - 0 (good)
# print(predictions2)  # Gives the class predicted - 1 (bad)

# Multiclass problem
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # No softmax at the end
        return out


model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss()  # Applies Softmax

# Binary classification
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # sigmoid at the end
        y_pred = torch.sigmoid(out)
        return y_pred


model = NeuralNet1(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss()  # Applies Softmax

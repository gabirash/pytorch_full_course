# 1) Design model (input size, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop:
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from triton.language import tensor

# import matplotlib.pyplot as plt


# 0) Prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale features
sc = StandardScaler()  # Zero mean
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)  # Number of values over 1 column
y_test = y_test.view(y_test.shape[0], 1)  # Number of values over 1 column

# 1) Model - In case of linear regression - just one layer
# f = wx + b --> sigmoid at the end
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)  # We only want to have one value at the end

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))  # Will return a value between [0, 1]
        return y_predicted

model = LogisticRegression(n_features)

# 2) Loss and Optimizer
learning_rate = 0.01
criterion = nn.BCELoss()  # Binary cross-entropy
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    # Backward pass
    loss.backward()

    # Update
    optimizer.step()

    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')

def calc_accuracy(y_test: tensor):
    """
    Count the correct ones and divide by the total number of features.
    :return: accuracy
    """
    return y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])


with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()  # Need to replicate the sigmoid behaviour
    acc = calc_accuracy(y_test)
    print(f'accuracy = {acc:.4f}')

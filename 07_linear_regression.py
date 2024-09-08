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
import matplotlib.pyplot as plt


# 0) Prepare data
X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X_torch = torch.from_numpy(X_numpy.astype(np.float32))
Y_torch = torch.from_numpy(Y_numpy.astype(np.float32))
Y_torch = Y_torch.view(Y_torch.shape[0], 1)  # Number of values over 1 column

n_samples, n_features = X_torch.shape

# 1) Model - In case of linear regression - just one layer
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)  # Linear layer

# 2) Loss and Optimizer
learning_rate = 0.01
criterion = nn.MSELoss()  # Callable function (type of variable)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass and loss
    y_predicted = model(X_torch)
    loss = criterion(y_predicted, Y_torch)

    # Backward pass
    loss.backward()

    # Update
    optimizer.step()

    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')

# Plot
predicted = model(X_torch).detach().numpy()  # detach to get rid of calculate_grads=True
plt.plot(X_numpy, Y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()

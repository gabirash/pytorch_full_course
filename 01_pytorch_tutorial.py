import torch
import numpy as np


# Can use rand; zeros or ones as well...
x = torch.empty(1)  # Tensor with dimension 1 is a scalar
y = torch.empty(3)  # Tensor with dimension 1 and 3 elements
z = torch.empty(2, 3)  # Tensor which is a 2X3 matrix
a = torch.ones(2, 2, 3)  # Tensor which is a 2X2X3 matrix
# print(a.dtype)

a = torch.ones(2, 2, 3, dtype=torch.double)  # Tensor which is a 2X2X3 matrix
# print(a.dtype)
# print(a.size())

b = torch.tensor([2.5, 0.1])
# print(b)

x = torch.rand(2, 2)
y = torch.rand(2, 2)
z = x + y
q = torch.add(x, y)

# print(x)
# print(y)
# print(z)
# print(q)

x = torch.rand(4, 4)
print(x[1, 1].item())

# Reshape
y = x.view(-1, 8)
print(y.size())

a = torch.ones(5)
print(a)
b = a.numpy()
print(type(b))

c = np.ones(5)
d=torch.from_numpy(c)
print(d)

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)  # Can also .to("cpu")...
    z = x + y
    # print(z)

# requires_grad allows to later optimize this variable by calculating it gradients...
x = torch.ones(5, requires_grad=True)  # False by default...
print(x)

import torch


x = torch.rand(3, requires_grad=True)
# print(x)

y = x + 2
# print(y)  # grad_fn=<AddBackward0>

z = y * y * 2
# print(z)  # grad_fn=<MulBackward0>

# z = z.mean()
# print(z)

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
# z.backward(v)  # dz/dx --> Jacobian * Vector
# print(x.grad)

# Stop pytorch from creating the gradient functions (grad_fn):
# 1. x.requires_grad(False)
# 2. x.detach()
# 3. with torch.no_grad(): ...

weights = torch.ones(4, requires_grad=True)

for epoch in range(2):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)  # The gradients will be accumulated to the .grad attribute... Which is incorrect!

    weights.grad.zero_()  # Thus we clean the .grad attribute (tail_ is for updating weights)

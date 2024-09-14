# """methods"""
import torch
import torch.nn as nn
from torch.distributed.pipeline.sync.checkpoint import checkpoint


# torch.save(arg, PATH)  # Use tensors, models or dicts for saving (using pickle - serialised result)
#
# torch.load(PATH)
#
# model.load_state_dict(arg)
#
# '''save state_dict --> just the parameters'''
# torch.save(model.state_dict(), PATH)
# model.eval()
#
# # to load, model must be created again with parameters
# model = Model(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()

class Model(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = Model(n_input_features=6)
# train your model

# FILE = "model.pth"  # short for pytorch
# # torch.save(model, FILE)
# # model = torch.load(FILE)
# # model.eval()
# # torch.save(model.state_dict(), FILE)
# loaded_model = Model(n_input_features=6)
# loaded_model.load_state_dict(torch.load(FILE))
# model.eval()
#
# for param in model.parameters():
#     print(param)

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(optimizer.state_dict())  # Dictionary with the optimizer's attributes like the lr, momentum, etc...

# Let's say we want to stop in some point during the training and save a checkpoint

# checkpoint = {
#     "epoch": 90,
#     "model_state": model.state_dict(),
#     "optim_state": optimizer.state_dict()
# }
#
# torch.save(checkpoint, "checkpoint.pth")

loaded_ckpt = torch.load("checkpoint.pth")
epoch = loaded_ckpt["epoch"]

model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0)  # the lr will eventually be loaded as 0.01

model.load_state_dict(loaded_ckpt["model_state"])
optimizer.load_state_dict(loaded_ckpt["optim_state"])

print(optimizer.state_dict())

# Save on GPU, Load on CPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

device = torch.device("cpu")
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))

# Save on GPU, Load on GPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)

# Save on CPU, Load on GPU
torch.save(model.state_dict(), PATH)

device = torch.device("cuda")
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))
model.to(device)

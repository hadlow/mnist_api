import torch

model = torch.load('./mnist_cnn.pt')

print("Model's state_dict:")
for param_tensor in model:
    print(param_tensor, "\t", model[param_tensor].size())


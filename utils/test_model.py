import torch
from model import SimpleCNN

model = SimpleCNN()
print(model)

# Create a fake image batch: 4 images, 3 color channels, 32x32 pixels
x = torch.randn(4, 3, 32, 32)
y = model(x)
print("Output shape:", y.shape)

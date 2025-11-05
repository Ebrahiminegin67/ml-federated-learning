import torch, torchvision
import flwr as fl
import matplotlib
import numpy as np

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Flower:", fl.__version__)
print("Matplotlib:", matplotlib.__version__)
print("NumPy:", np.__version__)

cuda = torch.cuda.is_available()
print("CUDA available:", cuda)
if cuda:
    print("GPU name:", torch.cuda.get_device_name(0))

# CIFAR-10 check
from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor()])
ds_train = datasets.CIFAR10(root="./runs/data", train=True, download=True, transform=transform)
ds_test  = datasets.CIFAR10(root="./runs/data", train=False, download=True, transform=transform)
print("CIFAR-10:", len(ds_train), "train /", len(ds_test), "test samples")
print("OK!")

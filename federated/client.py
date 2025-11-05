import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import flwr as fl
from utils.model import SimpleCNN
from utils.train_test import train, test
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import csv


# ----- Split CIFAR-10 dataset into subsets -----
def load_datasets(num_clients=3, client_id=0):
    print(f" Loading data for client {client_id}...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_train = datasets.CIFAR10(root="./runs/data", train=True, download=True, transform=transform)
    full_test = datasets.CIFAR10(root="./runs/data", train=False, download=True, transform=transform)

    # Non-IID split: randomly assign different classes to each client
    random.seed(42)
    indices = np.arange(len(full_train))
    np.random.shuffle(indices)
    split_size = len(indices) // num_clients
    subset_indices = indices[client_id * split_size:(client_id + 1) * split_size]
    subset = Subset(full_train, subset_indices)

    train_loader = DataLoader(subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(full_test, batch_size=64, shuffle=False)

    return train_loader, test_loader


OPTIMIZER_TYPE = "Adam"  # Options: "Adam", "SGD"
LEARNING_RATE = 0.001   
LOCAL_EPOCHS = 3
BATCH_SIZE = 64

# ----- Define Client -----
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        if OPTIMIZER_TYPE == "Adam":
            self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        elif OPTIMIZER_TYPE == "SGD":
            self.optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        for epoch in range(LOCAL_EPOCHS):
            train_loss, train_acc = train(self.model, self.train_loader, self.criterion, self.optimizer, self.device)
            print(f" Client training | Epoch {epoch+1}/{LOCAL_EPOCHS} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        test_loss, test_acc = test(self.model, self.test_loader, self.criterion, self.device)

        log_results(f"Client_{id(self)}", test_acc, test_loss)

        return float(test_loss), len(self.test_loader.dataset), {"accuracy": float(test_acc)}



def start_client(client_id, server_address="127.0.0.1:8081", num_clients=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    print(f" Using device: {device}")
    train_loader, test_loader = load_datasets(num_clients, client_id)
    print(" Data loaded.")
    print(f" Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
    print(" Initializing client...")
    client = CifarClient(model, train_loader, test_loader, device)
    print(" Connecting to server...")
    fl.client.start_numpy_client(server_address=server_address, client=client)
    print(" Client finished.")


def log_results(config_name, test_acc, test_loss):
    os.makedirs("../runs/logs", exist_ok=True)
    log_path = "../runs/logs/results_log.csv"

    file_exists = os.path.isfile(log_path)
    with open(log_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Config", "Local Epochs", "LR", "Batch Size", "Optimizer", "Test Acc", "Test Loss"])
        writer.writerow([config_name, LOCAL_EPOCHS, LEARNING_RATE, BATCH_SIZE, OPTIMIZER_TYPE, test_acc, test_loss])

    
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        client_id = int(sys.argv[1])
    else:
        client_id = 0 

    print(f" Client starting with ID {client_id}...")
    try:
        start_client(client_id=client_id, server_address="127.0.0.1:8081", num_clients=3)
        print("Connected successfully!")
    except Exception as e:
        print("Connection failed:", e)


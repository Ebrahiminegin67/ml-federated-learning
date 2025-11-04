# Federated Learning with Flower (CIFAR-10)

This project demonstrates a Federated Learning (FL) framework using the Flower (FLWR) library in combination with PyTorch for training on the CIFAR-10 dataset. The implementation simulates multiple clients collaboratively training a convolutional neural network (CNN) model without sharing their raw data, showcasing the fundamental concepts of privacy-preserving distributed learning.

---

## 1. Overview

Federated Learning (FL) enables model training across multiple decentralized devices or servers holding local data samples, without exchanging the data itself. This project applies the **Federated Averaging (FedAvg)** strategy to aggregate model parameters from multiple clients, resulting in an improved global model.

The implementation consists of:
- A **server** that manages the aggregation of model updates.
- Multiple **clients** that train locally on different data subsets of the CIFAR-10 dataset.

---

## 2. Requirements and Installation

### 2.1 Clone the Project
```bash
git clone https://github.com/Ebrahiminegin67/ml-federated-learning.git
```
Redirect to the folder
```bash
cd ml-federated-learning
```

### 2.2. Virtual Environment Setup

It is recommended to use a Python virtual environment for isolated dependency management.

```bash
python -m venv venv
venv\Scripts\activate          # On Windows
# source venv/bin/activate      # On macOS/Linux
```

### 2.3. Install Dependencies

All dependencies required to run this project are listed in `requirements.txt` and can be installed using:

```bash
pip install -r requirements.txt
```

**requirements.txt includes:**
```
torch
torchvision
numpy
matplotlib
tqdm
flwr
scikit-learn
```

---

## 3. How to Execute the Program

### 3.1. Start the Server

To initiate the Federated Learning server, execute the following command from the root directory:

First activate the venv
```bash
venv\Scripts\activate          # On Windows
# source venv/bin/activate      # On macOS/Linux
```

```bash
python federated/server.py
```

### 3.2. Start the Clients

In separate terminal windows, start three clients as follows:

#### Client #1

**It is important to run all the clients**

Redirect the project folder

First activate the venv
```bash
venv\Scripts\activate          # On Windows
# source venv/bin/activate      # On macOS/Linux
```
and then run the client #1
```bash
python federated/client.py 0
```
#### Client #2
Redirect the project folder

First activate the venv
```bash
venv\Scripts\activate          # On Windows
# source venv/bin/activate      # On macOS/Linux
```
and then run the client #2
```bash
python federated/client.py 1
```
#### Client #3
Redirect the project folder

First activate the venv
```bash
venv\Scripts\activate          # On Windows
# source venv/bin/activate      # On macOS/Linux
```
and then run the client #3
```bash
python federated/client.py 2
```

Each client will load its assigned data partition, train a local CNN model, and send the learned parameters to the central server after each training round.

---

## 4. Project Structure and Code Description

```
├── federated/
│   ├── server.py       # Configures and runs the Flower server
│   ├── client.py       # Defines client logic and manages local training
│
├── utils/
│   ├── model.py        # Contains SimpleCNN model architecture
│   ├── train_test.py   # Defines training and testing procedures
│
├── requirements.txt    # List of required packages
```

### 4.1. Server Component (`server.py`)

- Initializes the Flower server with the **FedAvg** aggregation strategy.
- Defines a custom metric aggregation function to compute the average accuracy across all clients.
- Coordinates multiple training rounds (default: 3).

### 4.2. Client Component (`client.py`)

- Loads the CIFAR-10 dataset and splits it into non-identical partitions (Non-IID).
- Defines a **SimpleCNN** model for image classification.
- Implements a Flower `NumPyClient` for local training, parameter updates, and evaluation.
- Communicates with the central server via the Flower API to participate in global training.

---

## 5. Expected Output (First Run)

### 5.1. Server Output

```bash
Log: Starting Flower server...
INFO flwr  Starting server on 127.0.0.1:8081
INFO flwr  Round 1: fit (3 clients)
INFO flwr  Round 1: evaluate
INFO flwr  Round 2: fit (3 clients)
INFO flwr  Round 2: evaluate
INFO flwr  Round 3: fit (3 clients)
INFO flwr  Round 3: evaluate
INFO flwr  Final aggregated metrics: {'accuracy': 0.45}
```

### 5.2. Client Output

At the first run, the client starts to download the dataset

```bash
Client starting with ID 0...
 Using device: cpu
 Loading data for client 0...
100%|███████████████████████████████████████████████████████████████████████████████| 170M/170M [00:08<00:00, 20.7MB/s]
```

and it will be continued by this
```bash
 Data loaded.
 Train samples: 16666, Test samples: 10000
 Initializing client...
 Connecting to server...
 #### some deprication warning
 
INFO :
INFO :      Received: get_parameters message de6f3cce-9d80-4007-bc9a-208077d43b42
INFO :      Sent reply
```

Outputs for clients `1` and `2` are similar, with unique subsets of training data.

---

## 6. Conclusion

This implementation demonstrates the core principles of Federated Learning using the Flower framework. By distributing model training across multiple clients, it preserves data privacy while collaboratively improving model performance. The experiment provides a foundational understanding of how decentralized machine learning systems can be implemented and tested locally.

---

**Author:** Negin Ebrahimi  
**Purpose:**  Assessment Project  
**Date:** 2025

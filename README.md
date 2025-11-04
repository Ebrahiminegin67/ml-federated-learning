# Federated Learning with Flower (CIFAR-10)

This project demonstrates a Federated Learning (FL) framework using the Flower (FLWR) library in combination with PyTorch for training on the CIFAR-10 dataset. The implementation simulates multiple clients collaboratively training a convolutional neural network (CNN) model without sharing their raw data, showcasing the fundamental concepts of privacy-preserving distributed learning.

---

## Table of Contents 
<!-- no toc -->
1. [Overview](#1-overview)  
2. [Requirements and Installation](#2-requirements-and-installation)  
3. [How to Execute the Program](#3-how-to-execute-the-program)  
   - [Start the Server](#31-start-the-server)  
   - [Start the Clients](#32-start-the-clients)  
4. [Project Structure and Code Description](#4-project-structure-and-code-description)   
5. [Expected Output (First Run)](#5-expected-output-first-run)  
   - [Server Output](#51-server-output)  
   - [Client Output](#52-client-output)  
6. [Conclusion](#6-conclusion)  


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

**It is highly important to run all the clients**

#### Client #1



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
INFO :      Starting Flower server, config: num_rounds=3, no round_timeout
INFO :      Flower ECE: gRPC server running (3 rounds), SSL is disabled
INFO :      [INIT]
INFO :      Requesting initial parameters from one random client
INFO :      Received initial parameters from one random client
INFO :      Starting evaluation of initial global parameters
INFO :      Evaluation returned no results (`None`)
INFO :
INFO :      [ROUND 1]
INFO :      configure_fit: strategy sampled 3 clients (out of 3)
INFO :      aggregate_fit: received 3 results and 0 failures
WARNING :   No fit_metrics_aggregation_fn provided
INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
INFO :      aggregate_evaluate: received 3 results and 0 failures
INFO :
INFO :      [ROUND 2]
INFO :      configure_fit: strategy sampled 3 clients (out of 3)
INFO :      aggregate_fit: received 3 results and 0 failures
INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
INFO :      aggregate_evaluate: received 3 results and 0 failures
INFO :
INFO :      [ROUND 3]
INFO :      configure_fit: strategy sampled 3 clients (out of 3)
INFO :      aggregate_fit: received 3 results and 0 failures
INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
INFO :      aggregate_evaluate: received 3 results and 0 failures
INFO :
INFO :      [SUMMARY]
INFO :      Run finished 3 round(s) in 777.69s
INFO :          History (loss, distributed):
INFO :                  round 1: 1.554962158203125
INFO :                  round 2: 1.2263801097869873
INFO :                  round 3: 1.0633295774459839
INFO :          History (metrics, distributed, evaluate):
INFO :          {'accuracy': [(1, 50.26), (2, 57.12), (3, 61.89000000000001)]}
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

INFO :
INFO :      Received: get_parameters message de6f3cce-9d80-4007-bc9a-208077d43b42
INFO :      Sent reply
INFO :
INFO :      Received: train message 3373eed3-d237-4917-a2c0-e56731ed59b2
INFO :      Sent reply
INFO :
INFO :      Received: evaluate message 948a41ab-bb20-44c4-9d91-ae33531649f1
INFO :      Sent reply
INFO :
INFO :      Received: train message 7f8340f0-24c9-4663-abbf-93d43ed6fe9a
INFO :      Sent reply
INFO :
INFO :      Received: evaluate message 8018dc79-b425-47ec-b954-96a59be67f85
INFO :      Sent reply
INFO :
INFO :      Received: train message 740a9451-e68a-45b6-8762-9ba69190fd68
INFO :      Sent reply
INFO :
INFO :      Received: evaluate message 61721979-aae7-42c0-a297-9fd9fd2f9168
INFO :      Sent reply
INFO :
INFO :      Received: reconnect message 2eec9fd8-4d70-4eab-bf27-59b81551fc06
INFO :      Disconnect and shut down
 Client finished.
Connected successfully!

```

Outputs for clients `1` and `2` are similar, with unique subsets of training data.

---

## 6. Conclusion

This implementation demonstrates the core principles of Federated Learning using the Flower framework. By distributing model training across multiple clients, it preserves data privacy while collaboratively improving model performance. The experiment provides a foundational understanding of how decentralized machine learning systems can be implemented and tested locally.

---

**Author:** Negin Ebrahimi  
**Purpose:**  Assessment Project  
**Date:** 2025

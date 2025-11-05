import matplotlib.pyplot as plt

# --- Centralized results
centralized_epochs = [1, 2, 3]
centralized_acc = [54.65, 61.34, 64.28]  # from centralized model output
centralized_loss = [1.2671, 1.0792, 1.0090]

# --- Federated results
federated_rounds = [1, 2, 3]
federated_acc = [50.4, 59.27, 63.95]
federated_loss = [1.6142, 1.1626, 1.0258]

# --- Plot Accuracy ---
plt.figure(figsize=(8, 5))
plt.plot(centralized_epochs, centralized_acc, marker='o', label='Centralized Accuracy')
plt.plot(federated_rounds, federated_acc, marker='s', label='Federated Accuracy')
plt.xlabel('Epoch / Round')
plt.ylabel('Accuracy (%)')
plt.title('Centralized vs Federated Learning Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../runs/accuracy_comparison.png")
plt.show()

# --- Plot Loss ---
plt.figure(figsize=(8, 5))
plt.plot(centralized_epochs, centralized_loss, marker='o', label='Centralized Loss')
plt.plot(federated_rounds, federated_loss, marker='s', label='Federated Loss')
plt.xlabel('Epoch / Round')
plt.ylabel('Loss')
plt.title('Centralized vs Federated Learning Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../runs/loss_comparison.png")
plt.show()

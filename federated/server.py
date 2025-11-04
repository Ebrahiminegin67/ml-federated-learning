import flwr as fl

# ----- 1. Define custom metric aggregation -----
def weighted_average(metrics):
    """Aggregate evaluation metrics from clients."""
    accuracies = [m[1]["accuracy"] for m in metrics]
    return {"accuracy": sum(accuracies) / len(accuracies)}

# ----- 2. Define the aggregation strategy -----
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,        # all clients participate each round
    fraction_evaluate=1.0,   # all clients evaluate each round
    min_fit_clients=3,       # minimum clients for training
    min_evaluate_clients=3,  # minimum clients for evaluation
    min_available_clients=3, # total clients that must be connected
    evaluate_metrics_aggregation_fn=weighted_average,
)

print("Log: Starting Flower server...")

# ----- 3. Start the Flower server -----
fl.server.start_server(
    server_address="127.0.0.1:8081",
    config=fl.server.ServerConfig(num_rounds=3),  # number of federated rounds
    strategy=strategy,
)

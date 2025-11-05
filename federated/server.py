import flwr as fl
import time
import csv
import os

def weighted_average(metrics):
    """Aggregate evaluation metrics from clients."""
    accuracies = [m[1]["accuracy"] for m in metrics]
    return {"accuracy": sum(accuracies) / len(accuracies)}

def log_server_results(total_time):
    import csv, os
    os.makedirs("../runs/logs", exist_ok=True)
    log_path = "../runs/logs/results_log.csv"

    with open(log_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([])
        writer.writerow(["Total Training Time (s)", round(total_time, 2)])

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,        
    fraction_evaluate=1.0,   
    min_fit_clients=3,       
    min_evaluate_clients=3,  
    min_available_clients=3, 
    evaluate_metrics_aggregation_fn=weighted_average)
print("Log: Starting Flower server...")

start_time = time.time()

fl.server.start_server(
    server_address="127.0.0.1:8081",
    config=fl.server.ServerConfig(num_rounds=3), 
    strategy=strategy)

end_time = time.time()
total_time = end_time - start_time
print(f"Total training time: {total_time:.2f} seconds")
log_server_results(total_time)


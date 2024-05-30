from typing import List, Tuple
from flwr.server import ServerApp, ServerConfig
from flwr.common import Metrics
import flwr as fl
from model import create_model

class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.global_model = create_model(512, 5)  # Update input_shape and num_classes accordingly based on model
        self.num_rounds = kwargs.get('num_rounds', 3)  # edit the "3" to what round should the global model will be saved
        self.current_round = 0

    def aggregate_fit(self, rnd: int, results: List[Tuple[fl.common.Parameters, fl.common.FitRes]], failures: List[str]) -> Tuple[fl.common.Parameters, Metrics]:

        # Aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        
        # Save the global model at each round
        self.save_global_model(aggregated_parameters, rnd)
        
        return aggregated_parameters, aggregated_metrics

    def save_global_model(self, parameters, rnd):
        # Extract the actual weights from the Parameters object
        weights = fl.common.parameters_to_ndarrays(parameters)
        self.global_model.set_weights(weights)
        filename = f"C:/Users/aldri/OneDrive/Desktop/Federated-Face-Recognition/global model/global_model_round{rnd}.keras" # edit with specified file path "global_model_round{rnd}.keras" is file name
        self.global_model.save(filename)
        print(f"Global model saved as {filename}")
        
        # Save as final_global_model after the final round
        if rnd == self.num_rounds:
            final_filename = "C:/Users/aldri/OneDrive/Desktop/Federated-Face-Recognition/global model/global_model.keras" #edit with specified file path "final_global_model.keras" is file name
            self.global_model.save(final_filename)
            print(f"Final global model saved as {final_filename}")

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated = super().aggregate_evaluate(rnd, results, failures)
        accuracies = [res[1].metrics['accuracy'] for res in results if res[1].metrics is not None]
        if accuracies:
            average_accuracy = sum(accuracies) / len(accuracies)
            print(f"Round {rnd}: Average accuracy: {average_accuracy:.3f}")
            # Log individual accuracies for debugging
            for i, accuracy in enumerate(accuracies):
                print(f"Client {i+1} accuracy: {accuracy:.3f}")
        else:
            print(f"Round {rnd}: No accuracies to average.")
        return aggregated

    def select_clients(self, results):
        """Select a subset of clients for model update."""
        # Use a subset selection strategy (e.g., top-k clients based on accuracy)
        num_clients = len(results)
        num_select = min(max(1, int(num_clients * 0.8)), num_clients)  # Select top 80% of clients
        sorted_results = sorted(results, key=lambda x: x[1].metrics["accuracy"], reverse=True)
        selected_clients = sorted_results[:num_select]
        return selected_clients


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    if sum(examples) == 0:
        return {"accuracy": 0.0}
    return {"accuracy": sum(accuracies) / sum(examples)}

# Define strategy and specify the minimum available client and minimum number of client that is training before federated learning process initiate
strategy = CustomFedAvg(evaluate_metrics_aggregation_fn=weighted_average, min_available_clients=3, min_fit_clients=3) 

# Define config
config = ServerConfig(num_rounds=3) # Edit Rounds

# Flower ServerApp
app = ServerApp(
    config=config,
    strategy=strategy,
)

# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server

    start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
    )

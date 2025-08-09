import subprocess
import sys
import time
from typing import Tuple, List

import flwr as fl
import matplotlib.pyplot as plt

from FedStrategy import FedSGDStrategy, FedAvgStrategy
from init_model_weights import model_init_fn


SERVER_ADDRESS = "127.0.0.1:9091"
NUM_ROUNDS = 10
NUM_CLIENTS = 5


def spawn_clients_subprocess(num_clients: int, start_delay: float = 1.5) -> subprocess.Popen:
    # Spawn a helper script which starts clients after a short delay
    return subprocess.Popen([sys.executable, "client_spawner.py", str(num_clients), str(start_delay)])


def run_server_blocking(strategy: fl.server.strategy.Strategy):
    return fl.server.start_server(
        server_address=SERVER_ADDRESS,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    )


def run_round(strategy_name: str, strategy: fl.server.strategy.Strategy):
    print(f"Starting server for {strategy_name}...")
    # Spawn clients to connect shortly after server comes up
    spawner = spawn_clients_subprocess(NUM_CLIENTS, start_delay=2.0)
    history = run_server_blocking(strategy)
    # Ensure spawner finishes
    spawner.wait()
    return history


def extract_metric(history, key: str) -> Tuple[List[int], List[float]]:
    if key not in history.metrics_distributed:
        return [], []
    pairs = history.metrics_distributed[key]
    rounds = [r for r, _ in pairs]
    values = [v for _, v in pairs]
    return rounds, values


def main():
    print("=== Comparing FedSGD vs FedAvg ===")

    # FedSGD run
    print("\n[1/2] Running FedSGD...")
    sgd_strategy = FedSGDStrategy(model_init_fn=model_init_fn, lr=0.01)
    hist_sgd = run_round("FedSGD", sgd_strategy)

    # Small pause to ensure the port is free
    time.sleep(2)

    # FedAvg run
    print("\n[2/2] Running FedAvg...")
    favg_strategy = FedAvgStrategy(model_init_fn=model_init_fn)
    hist_favg = run_round("FedAvg", favg_strategy)

    # Plot comparison for common metrics
    metrics = [
        "accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
    ]

    plt.figure(figsize=(10, 8))
    subplot_idx = 1
    plotted_any = False
    for m in metrics:
        r1, v1 = extract_metric(hist_sgd, m)
        r2, v2 = extract_metric(hist_favg, m)
        if not r1 and not r2:
            continue
        plotted_any = True
        plt.subplot(2, 2, subplot_idx)
        if r1:
            plt.plot(r1, v1, label="FedSGD")
        if r2:
            plt.plot(r2, v2, label="FedAvg")
        plt.title(m)
        plt.xlabel("Round")
        plt.ylabel(m)
        plt.grid(True)
        plt.legend()
        subplot_idx += 1

    if plotted_any:
        plt.tight_layout()
        out_path = "comparison_metrics.png"
        plt.savefig(out_path)
        print(f"\nSaved comparison plot to {out_path}")
        try:
            plt.show()
        except Exception:
            # In headless envs, just skip showing
            pass
    else:
        print("No comparable metrics found in histories.")


if __name__ == "__main__":
    main()

import flwr as fl
import torch
from dataset import load_dataset, get_num_classes
from model import FNN
from FedStrategy import FedSGDStrategy
from init_model_weights import model_init_fn
import matplotlib.pyplot as plt

NUM_CLASSES = get_num_classes()

strategy = FedSGDStrategy(model_init_fn=model_init_fn, lr=0.01)

print("Starting Flower Server (FedSGD)...")

history = fl.server.start_server(server_address="127.0.0.1:9091", strategy=strategy, config=fl.server.ServerConfig(num_rounds=10))
# Sau khi huấn luyện xong
rounds = [r for r, _ in history.metrics_distributed["accuracy"]]
accuracy = [v for _, v in history.metrics_distributed["accuracy"]]
macro_f1 = [v for _, v in history.metrics_distributed["macro_f1"]]
macro_precision = [v for _, v in history.metrics_distributed["macro_precision"]]
macro_recall = [v for _, v in history.metrics_distributed["macro_recall"]]

plt.figure(figsize=(10,6))
plt.plot(rounds, accuracy, label="Accuracy")
plt.plot(rounds, macro_f1, label="Macro F1")
plt.plot(rounds, macro_precision, label="Macro Precision")
plt.plot(rounds, macro_recall, label="Macro Recall")
plt.xlabel("Round")
plt.ylabel("Metric Value")
plt.title("Federated Learning Metrics per Round")
plt.legend()
plt.grid(True)
plt.show()
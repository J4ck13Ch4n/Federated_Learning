import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from torch.utils.data import DataLoader
from dataset import load_dataset, get_num_classes, load_partition
from model import FNN
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = get_num_classes()
trainset, testset,_ = load_dataset()
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32)

input_size = trainset[0][0].shape[0]
model = FNN(input_size=input_size, hidden_size1=128, hidden_size2=64, num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training one epoch
def train_one_step():
    model.train()
    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    return loss.item()

# Evaluation
def test_model():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# Flower Client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, num_clients=10):
        self.cid = cid
        self.trainset, self.testset = load_partition(cid, num_clients=num_clients, noniid=True, alpha=0.5)
        self.model = model.to(device)
        self.testloader = testloader
        # In số record & phân bố class
        labels = [y.item() for _, y in self.trainset]
        from collections import Counter
        counts = Counter(labels)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nClient {cid} | Total samples: {len(labels)}")
        for cls, cnt in counts.items():
            print(f"Class {cls}: {cnt} ({cnt/len(labels):.2%})")
            
    def get_parameters(self, config):
        return [v.cpu().numpy() for v in model.state_dict().values()]

    def set_parameters(self, parameters):
        keys = list(model.state_dict().keys())
        state_dict = {k: torch.tensor(p) for k, p in zip(keys, parameters)}
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loss = train_one_step()
        return self.get_parameters(config), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        # Cập nhật model
        self.set_parameters(parameters)
        self.model.eval()

        y_true, y_pred = [], []
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for xb, yb in self.testloader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                outputs = self.model(xb)
                loss = criterion(outputs, yb)
                total_loss += loss.item() * xb.size(0)
                preds = outputs.argmax(1)
                y_true.extend(yb.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        # Tính metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

        acc = accuracy_score(y_true, y_pred)
        precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
        recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1s = f1_score(y_true, y_pred, average=None, zero_division=0)

        # Log per-class metric
        for i, (p, r, f) in enumerate(zip(precisions, recalls, f1s)):
            print(f"[Client {self.cid}] Class {i}: Precision={p:.4f} Recall={r:.4f} F1={f:.4f}")

        avg_loss = total_loss / len(self.testloader.dataset)

        # Trả về đúng format yêu cầu
        return avg_loss, len(self.testloader.dataset), {
            "accuracy": acc,
            "macro_precision": float(np.mean(precisions)),
            "macro_recall": float(np.mean(recalls)),
            "macro_f1": float(np.mean(f1s))
        }




# Start Client
if __name__ == "__main__":
    import sys
    cid = int(sys.argv[1])  # Lấy client ID từ command line
    fl.client.start_numpy_client(
        server_address="127.0.0.1:9091",
        client=FlowerClient(cid, num_clients=5)  #Truyền cid vào đây
    )


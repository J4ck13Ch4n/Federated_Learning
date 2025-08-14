import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout)
        # Lớp tuyến tính cuối cùng sẽ có đầu vào là hidden_size * 2 vì là bidirectional
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # LSTM yêu cầu input có dạng 3D: (batch_size, sequence_length, input_size)
        # Bạn cần reshape lại dữ liệu đầu vào trước khi đưa vào model
        # Ví dụ: x = x.view(x.size(0), 1, x.size(1)) 
        # ở đây sequence_length = 1, bạn cần điều chỉnh cho phù hợp
        
        # Khởi tạo hidden state và cell state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  
        
        # Lấy output của time step cuối cùng
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
import torch


class TorchLSTM(torch.nn.Module):
    def __init__(self, sequence_length, input_size, hidden_size, num_layers, num_classes, dropout_probs):
        """
        长短期记忆网络（Long Short-Term Memory, LSTM）

        :param sequence_length:
        :param input_size:
        :param hidden_size:
        :param num_layers:
        :param num_classes:
        :param dropout_probs:
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.dropout = torch.nn.Dropout(dropout_probs)
        self.out_linear = torch.nn.Sequential(
            torch.nn.Linear(sequence_length * hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_classes)
        )
        # 初始化
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, self.sequence_length, self.hidden_size)
        c0 = torch.zeros(self.num_layers, self.sequence_length, self.hidden_size)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        return torch.stack([self.out_linear(out[i].view(-1)) for i in range(out.shape[0])])

    def to(self, device):
        super().to(device)
        self.lstm = self.gru.to(device)
        self.out_linear = self.out_linear.to(device)
        return self

import torch


class TorchLSTM(torch.nn.Module):
    def __init__(self, sequence_length, embedding_dim, hidden_size, num_layers, num_classes,
                 dropout_probs, embedding=None):
        """
        长短期记忆网络（Long Short-Term Memory, LSTM）

        :param sequence_length:
        :param embedding_dim:
        :param hidden_size:
        :param num_layers:
        :param num_classes:
        :param dropout_probs:
        :param embedding:
        """
        super().__init__()
        self.embedding = embedding
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers)
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
        x = self.embedding.vectors[x]
        h0 = torch.zeros(self.num_layers, self.sequence_length, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, self.sequence_length, self.hidden_size).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        return torch.stack([self.out_linear(out[i].view(-1)) for i in range(out.shape[0])])

    def to(self, device):
        super().to(device)
        self.embedding = self.embedding.to(device)
        self.lstm = self.lstm.to(device)
        self.out_linear = self.out_linear.to(device)
        return self

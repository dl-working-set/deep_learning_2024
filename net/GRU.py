import torch


class TorchGRU(torch.nn.Module):
    def __init__(self, sequence_length, embedding_dim, hidden_size, num_layers, num_classes,
                 dropout_probs, embedding=None):
        """
        门控循环单元网络（Gated Recurrent Unit, GRU）

        :param sequence_length:
        :param embedding_dim:
        :param num_embeddings:
        :param hidden_size:
        :param num_layers:
        :param num_classes:
        :param dropout_probs:
        :param embedding:
        """
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(embedding.vectors)
        self.embedding.weight.requires_grad = True

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = torch.nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers)
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
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers, self.sequence_length, self.hidden_size).to(x.device)
        out, hn = self.gru(x, h0)
        out = self.dropout(out)
        return torch.stack([self.out_linear(out[i].view(-1)) for i in range(out.shape[0])])

    def to(self, device):
        super().to(device)
        self.embedding = self.embedding.to(device)
        self.gru = self.gru.to(device)
        self.out_linear = self.out_linear.to(device)
        return self

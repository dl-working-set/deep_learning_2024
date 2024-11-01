import torch


class AttentionBiLSTM(torch.nn.Module):
    def __init__(self, sequence_length, embedding_dim, hidden_size, num_layers, num_classes,
                 dropout_probs, embedding=None):
        """
        基于注意力的双向长短期记忆（Attention-Based Bidirectional Long Short-Term Memory）
        论文：https://aclanthology.org/P16-2034.pdf

        :param sequence_length:
        :param embedding_dim:
        :param hidden_size:
        :param num_layers:
        :param num_classes:
        :param dropout_probs:
        :param embedding:
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.embedding = torch.nn.Embedding.from_pretrained(embedding.vectors)
        self.embedding.weight.requires_grad = True

        self.lstm = torch.nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size, num_layers=num_layers,
                                  bidirectional=True, dropout=dropout_probs, batch_first=True)

        self.weight_W = torch.nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.weight_proj = torch.nn.Parameter(torch.Tensor(2 * hidden_size, 1))

        self.decoder1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, hidden_size),  # D=2 if bidirectional = True else 1
            torch.nn.LeakyReLU(inplace=True),
        )
        # 增加dropout层
        self.decoder2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, num_classes),
        )

        # 初始化
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)

    def forward(self, x):
        embeddings = self.embedding(x)

        # LSTM层
        # output shape: (batch, seq_len, hidden_size*num_directions方向数)
        # hidden[0/1] shape: (num_layers层数 * num_directions方向数, batch, hidden_size)
        H, _ = self.lstm(embeddings)

        # Attention层
        # 中间变量 M = tanh(H) => [batch, seq_len, hidden_size*num_directions]
        M = torch.tanh(torch.matmul(H, self.weight_W))
        # att shape: (seq_len, batch, 1) =>
        # 注意力分数 alpha = softmax(wT * M) => [batch_size, seq_len, 1]
        alpha = torch.nn.functional.softmax(torch.matmul(M, self.weight_proj), dim=1)
        # 加权后输出 r = H * alphaT => [batch, seq_len, hidden_size*num_directions]
        r = H * alpha

        # Classification层
        out = self.decoder1(torch.sum(r, dim=1))
        out = self.decoder2(out)
        return out

    def to(self, device):
        super().to(device)
        self.embedding = self.embedding.to(device)
        self.lstm = self.lstm.to(device)
        self.weight_W = self.weight_W.to(device)
        self.weight_proj = self.weight_proj.to(device)
        self.decoder1 = self.decoder1.to(device)
        self.decoder2 = self.decoder2.to(device)
        return self
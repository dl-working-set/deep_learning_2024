import torch


class TextCNN(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_classes,
                 dropout_probs, embedding=None, kernel_sizes=[3, 4, 5], num_channels=100):
        """
        TextCNN模型
        https://arxiv.org/pdf/1408.5882v2

        :param sequence_length:
        :param embedding_dim:
        :param hidden_size:
        :param num_layers:
        :param num_classes:
        :param dropout_probs:
        :param embedding:
        """
        super(TextCNN, self).__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(embedding.vectors)
        self.embedding.weight.requires_grad = True

        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=1, out_channels=num_channels, kernel_size=(k, embedding_dim))
            for k in kernel_sizes
        ])
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_probs)
        self.classification = torch.nn.Linear(len(kernel_sizes) * num_channels, num_classes)  # shape: [300, 5]

        # 初始化
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)

    def forward(self, x):
        # (batch_size, seq_len, embed_dim)
        x = self.embedding(x)
        # (batch_size, channel=1, seq_len, embed_dim)
        x = x.unsqueeze(1)
        # [(batch_size, num_channels, seq_len - k + 1) for k in kernel_sizes]
        x = [self.activation(conv(x)).squeeze(3) for conv in self.convs]
        # [(batch_size, num_channels) for k in kernel_sizes]
        x = [torch.nn.functional.max_pool1d(c, c.size(2)).squeeze(2) for c in x]
        # (batch_size, len(kernel_sizes) * num_channels)
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        # (batch_size, num_classes)
        x = self.classification(x)
        return x

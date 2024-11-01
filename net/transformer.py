import math

import torch


class ClassificationHead(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc1 = torch.nn.Linear(embedding_dim, hidden_size)  # 中间层
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

        # 初始化
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)

    def forward(self, x):
        # 取序列的第一个位置作为分类输入（也可以取平均或最大池化）
        # x = x[:, 0, :]  # [batch_size, embedding_dim]
        x = x.max(dim=1)[0]  # [batch_size, embedding_dim]
        x = self.fc1(x.view(x.shape[0], -1))  # [batch_size, hidden_size]
        x = self.norm(x)  # [batch_size, hidden_size]
        x = self.relu(x)  # [batch_size, hidden_size]
        x = self.fc2(x)  # [batch_size, num_classes]
        return x


class PositionalEncoding(torch.nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads, dim_feedforward=2048, dropout_probs=0.):
        super().__init__()
        self.self_attention = torch.nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads,
                                                          dropout=dropout_probs)
        self.norm1 = torch.nn.LayerNorm(embedding_dim)

        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, dim_feedforward),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_probs),
            torch.nn.Linear(dim_feedforward, embedding_dim),
            torch.nn.Dropout(dropout_probs),
        )
        self.norm2 = torch.nn.LayerNorm(embedding_dim)

        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Multi-Head Self-Attention
        sa_out, attn_weights = self.self_attention(src, src, src, attn_mask=src_mask,
                                                   key_padding_mask=src_key_padding_mask)
        src = sa_out + src  # Residual Connection
        src = self.norm1(src)  # 归一化

        # Feed-Forward Network
        ff_out = self.feed_forward(src)
        src = ff_out + src  # Residual Connection
        src = self.norm2(src)  # 归一化

        return src


class TransformerEncoder(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_heads, dim_feedforward, nlayers, num_classes,
                 dropout_probs=0.):
        super(TransformerEncoder, self).__init__()
        self.model_type = 'TransformerEncoder'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(embedding_dim=embedding_dim)
        self.encoder = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.layers = torch.nn.ModuleList(
            [TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward, dropout_probs) for _ in range(nlayers)])
        self.embedding_dim = embedding_dim
        self.classifier = ClassificationHead(embedding_dim=embedding_dim, hidden_size=dim_feedforward,
                                             num_classes=num_classes)
        # 初始化
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)

    def forward(self, src):
        src_mask = None  # Decoder
        src_key_padding_mask = (src == 0)
        # Embedding and Positional Encoding
        # src = self.encoder(src) * math.sqrt(self.embedding_dim)
        src = self.encoder(src)
        src = self.pos_encoder(src)

        # Pass through the encoder layers
        # Convert from [batch_size, seq_len, embedding_dim] to [seq_len, batch_size, embedding_dim]
        output = src.transpose(0, 1)  # shape: [seq_len, batch_size, embedding_dim]
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # Convert back to [batch_size, seq_len, embedding_dim]
        output = output.transpose(0, 1)

        # Classification Head:128,config.SEQUENCE_LENGTH,100 * 100,6
        output = self.classifier(output)
        # Convert back to [batch_size, seq_len, embedding_dim]
        return output

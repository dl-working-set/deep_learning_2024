import math

import torch

from init import config


class ClassificationHead(torch.nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # 取序列的第一个位置作为分类输入（也可以取平均或最大池化）
        x = x[:, 0, :]  # [batch_size, embedding_dim]
        x = self.fc(x)  # [batch_size, num_classes]
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
        self.self_attn = torch.nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_probs)
        self.linear1 = torch.nn.Linear(embedding_dim, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout_probs)
        self.linear2 = torch.nn.Linear(dim_feedforward, embedding_dim)

        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.norm2 = torch.nn.LayerNorm(embedding_dim)
        self.dropout1 = torch.nn.Dropout(dropout_probs)
        self.dropout2 = torch.nn.Dropout(dropout_probs)

        self.activation = torch.nn.ReLU()
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Multi-Head Self-Attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed-Forward Network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(torch.nn.Module):
    def __init__(self, embedding_dim=100, num_heads=5, dim_feedforward=1024, nlayers=6, num_classes=5,
                 dropout_probs=0., embedding=None):
        super(TransformerEncoder, self).__init__()
        self.model_type = 'TransformerEncoder'
        self.src_mask = None
        # self.pos_encoder = PositionalEncoding(embedding_dim=embedding_dim, max_len=num_embeddings)
        self.pos_encoder = PositionalEncoding(embedding_dim=embedding_dim)
        # 使用预训练gensim.word2vec 代替 torch.nn.Embedding
        # self.encoder = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.encoder = embedding
        self.layers = torch.nn.ModuleList(
            [TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward, dropout_probs) for _ in range(nlayers)])
        self.embedding_dim = embedding_dim
        self.classifier = ClassificationHead(embedding_dim, num_classes)
        # 初始化
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)

    def forward(self, src):
        src_mask = None  # Decoder

        # 使用 <PAD> 标记mask
        # src_key_padding_mask = (src == 0)
        src_key_padding_mask = (src == self.encoder.key_to_index[config.padding_word])

        # Embedding and Positional Encoding
        # src = self.encoder.vectors[src] * math.sqrt(self.embedding_dim)
        src = self.encoder.vectors[src]
        # 应用位置编码
        src = self.pos_encoder(src)

        # Pass through the encoder layers
        # Convert from [batch_size, seq_len, embedding_dim] to [seq_len, batch_size, embedding_dim]
        output = src.transpose(0, 1)
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # Convert back to [batch_size, seq_len, embedding_dim]
        output = output.transpose(0, 1)

        # Classification Head:128,config.SEQUENCE_LENGTH,100 * 100,6
        output = self.classifier(output)
        # Convert back to [batch_size, seq_len, embedding_dim]
        return output

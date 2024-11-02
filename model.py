"""
- 模型搭建
  - 模型搭建
    - CNN or RNN
    - 输出层：5分类
  - 模型保存
"""
import os

import torch

from net.GRU import TorchGRU
from net.attention_bi_lstm import AttentionBiLSTM
from net.textCNN import TextCNN
from net.transformer import TransformerEncoder


class SentimentAnalysisModel(torch.nn.Module):
    def __init__(self, model_type=None,
                 sequence_length=100,
                 embedding_dim=100,
                 hidden_size=1024,
                 num_classes=5,
                 dropout_probs=0.,
                 embedding=None, ):
        """
        情绪分析模型

        :param model_type:
        :param sequence_length:
        :param embedding_dim:
        :param hidden_size:
        :param num_classes:
        :param dropout_probs:
        :param embedding:
        """
        super().__init__()
        if model_type is None:
            return

        self.model_type = model_type
        self.net = None  # 具体网络实现 ⬇️

        # 模型类型：textCNN、GRU、attention_bi_lstm、transformer
        if self.model_type == 'textCNN':
            self.net = TextCNN(embedding_dim=embedding_dim, hidden_size=hidden_size, num_classes=num_classes,
                               dropout_probs=dropout_probs, embedding=embedding, kernel_sizes=[3, 4, 5],
                               num_channels=100)
        if self.model_type == 'GRU':
            self.net = TorchGRU(sequence_length=sequence_length, embedding_dim=embedding_dim,
                                hidden_size=hidden_size, num_layers=1, num_classes=num_classes,
                                dropout_probs=dropout_probs, embedding=embedding)
        elif self.model_type == 'attention_bi_lstm':
            self.net = AttentionBiLSTM(sequence_length=sequence_length, embedding_dim=embedding_dim,
                                       hidden_size=hidden_size, num_layers=2, num_classes=num_classes,
                                       dropout_probs=dropout_probs, embedding=embedding)
        elif self.model_type == 'transformer':
            self.net = TransformerEncoder(embedding_dim=embedding_dim, dim_feedforward=hidden_size,
                                          nlayers=6, num_heads=4, num_classes=num_classes, dropout_probs=dropout_probs,
                                          embedding=embedding)
        elif self.model_type == '*':
            pass

    def forward(self, x):
        return self.net(x)

    def to(self, device):
        super().to(device)
        self.net = self.net.to(device)
        return self

    def save(self, filename):
        path = os.path.dirname(__file__)
        absolute_path = os.path.join(path, filename)
        torch.save({
            'model_type': self.model_type,
            'net': self.net,
            'model_state_dict': self.state_dict(),
        }, absolute_path)

    @classmethod
    def load(cls, filename):
        path = os.path.dirname(__file__)
        absolute_path = os.path.join(path, filename)
        checkpoint = torch.load(f=absolute_path, weights_only=False)
        self = cls()
        self.model_type = checkpoint['model_type']
        self.net = checkpoint['net']
        self.load_state_dict(checkpoint['model_state_dict'])
        self.eval()
        return self

"""
- 模型搭建
  - 模型搭建
    - CNN or RNN
    - 输出层：5分类
  - 模型保存
"""
import torch

from net.GRU import TorchGRU
from net.LSTM import TorchLSTM
from net.transformer import TransformerEncoder


class SentimentAnalysisModel(torch.nn.Module):
    def __init__(self, model_type=None,
                 sequence_length=100,
                 embedding_dim=100,
                 num_embeddings=10000,
                 hidden_size=1024,
                 num_layers=1,
                 output_size=6,
                 dropout_probs=0.):
        """
        情绪分析模型

        :param model_type: 
        :param sequence_length: 
        :param embedding_dim: 
        :param num_embeddings: 
        :param hidden_size: 
        :param num_layers: 
        :param output_size: 
        :param dropout_probs: 
        """
        super().__init__()
        if model_type is None:
            return

        self.model_type = model_type
        self.net = None  # 具体网络实现 ⬇️

        # 模型类型：GRU、LSTM、Transformer、
        if self.model_type == 'GRU':
            self.net = TorchGRU(sequence_length=sequence_length, embedding_dim=embedding_dim,
                                num_embeddings=num_embeddings, hidden_size=hidden_size,
                                num_layers=num_layers, output_size=output_size, dropout_probs=dropout_probs)
        elif self.model_type == 'LSTM':
            self.net = TorchLSTM(sequence_length=sequence_length, embedding_dim=embedding_dim,
                                 num_embeddings=num_embeddings, hidden_size=hidden_size,
                                 num_layers=num_layers, output_size=output_size, dropout_probs=dropout_probs)
        elif self.model_type == 'Transformer':
            self.net = TransformerEncoder(num_embeddings=num_embeddings, embedding_dim=embedding_dim, num_heads=4,
                                          dim_feedforward=hidden_size, nlayers=6, num_classes=output_size,
                                          dropout_probs=dropout_probs)
        elif self.model_type == '*':
            pass

    def forward(self, x):
        return self.net(x)

    def to(self, device):
        super().to(device)
        self.net = self.net.to(device)
        return self

    def save(self, filename):
        torch.save({
            'model_type': self.model_type,
            'net': self.net,
            'model_state_dict': self.state_dict(),
        }, filename)

    @classmethod
    def load(self, filename):
        checkpoint = torch.load(filename)
        model = self()
        model.model_type = checkpoint['model_type']
        model.net = checkpoint['net']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

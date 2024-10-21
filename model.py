"""
- 模型搭建
  - 模型搭建
    - CNN or RNN
    - 输出层：5分类
  - 模型保存
"""
import torch


class SentimentAnalysisModel(torch.nn.Module):
    """情绪分析模型：GRU+全连接

    Args:
        input_size: 输入层通道数
        sequence_length: 序列长度
        hidden_size: 隐藏层通道数
        num_layers: 网络层数
        output_size: 输出分类数

    """

    def __init__(self, input_size, sequence_length=100, hidden_size=1024, num_layers=1, output_size=6):
        super().__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.gru = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.h0 = torch.zeros(self.num_layers, self.sequence_length, self.hidden_size)
        self.out_linear = torch.nn.Sequential(
            torch.nn.Linear(sequence_length * hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )
        # 初始化
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)

    def forward(self, x):
        out, hn = self.gru(x, self.h0)
        return torch.stack([self.out_linear(out[i].view(-1)) for i in range(out.shape[0])]), hn

    def save(self, filename):
        torch.save({
            'input_size': self.input_size,
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'gru': self.gru,
            'h0': self.h0,
            'out_linear': self.out_linear,
            'model_state_dict': self.state_dict(),
        }, filename)

    @classmethod
    def load(self, filename):
        checkpoint = torch.load(filename)
        model = self()
        model.input_size = checkpoint['input_size']
        model.sequence_length = checkpoint['sequence_length']
        model.hidden_size = checkpoint['hidden_size']
        model.gru = checkpoint['gru']
        model.h0 = checkpoint['h0']
        model.out_linear = checkpoint['out_linear']
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

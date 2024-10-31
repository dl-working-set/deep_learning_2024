# !usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False  


class LSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        pretrained_weight,
        update_w2v,
        hidden_dim,
        num_layers,
        drop_keep_prob,
        n_class,
        bidirectional,
        **kwargs
    ):
        """
        @description: initial seetings
        @param {*}
        - vocab_size: int, vocabulary size.
        - embedding_dim: int, the embedding layer dim.
        - pretrained_weight: Tensor, FloatTensor containing weights for the Embedding.
        - update_w2v: bool, whether to update word2vec embedding weight.
        - hidden_dim: int, the hidden layer dim.
        - num_layers: int, the number of layers.
        - drop_keep_prob: float, the keep probability of dropout layer.
        - n_class: int, the number of classes (labels).
        - bidirectional: bool, whether to use bidirectional LSTM.
        @return {*}
        None
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_class = n_class

        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(pretrained_weight)
        self.embedding.weight.requires_grad = update_w2v
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            bidirectional=self.bidirectional,
            dropout=drop_keep_prob,
        )

        if self.bidirectional:
            self.decoder1 = nn.Linear(hidden_dim * 4, hidden_dim)
            self.decoder2 = nn.Linear(hidden_dim, n_class)
        else:
            self.decoder1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.decoder2 = nn.Linear(hidden_dim, n_class)

    def forward(self, inputs):
        # inputs: (num_embeddings, embedding_dim) => [batch, seq_len, embed_dim]:[64,75,50]
        embeddings = self.embedding(inputs)
        # LSTM input:(seq, batch, input_size) => [seq_len, batch, embed_dim]:[75,64,50]
        # 在 PyTorch 中，permute 方法用于改变张量的维度顺序
        states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
        # states: (seq_len, batch, D*hidden_size), D=2 if bidirectional = True else 1, =>[75,64,256]
        # hidden: (h_n, c_n) => h_n / c_n shape:(D∗num_layers, batch, hidden_size) =>[4,64,128]
        encoding = torch.cat([states[0], states[-1]], dim=1)
        # encoding shape: (batch, 2*D*hidden_size): [64,512]
        outputs = self.decoder1(encoding)
        # outputs = F.softmax(outputs, dim=1)
        outputs = self.decoder2(outputs)  # outputs shape:(batch, n_class) => [64,2]
        return outputs


class LSTM_attention(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        pretrained_weight,
        update_w2v,
        hidden_dim,
        num_layers,
        drop_keep_prob,
        n_class,
        bidirectional,
        **kwargs
    ):
        """
        @description: initial seetings
        @param {*}
        - vocab_size: int, vocabulary size.
        - embedding_dim: int, the embedding layer dim.
        - pretrained_weight: Tensor, FloatTensor containing weights for the Embedding.
        - update_w2v: bool, whether to update word2vec embedding weight.
        - hidden_dim: int, the hidden layer dim.
        - num_layers: int, the number of layers.
        - drop_keep_prob: float, the keep probability of dropout layer.
        - n_class: int, the number of classes (labels).
        - bidirectional: bool, whether to use bidirectional LSTM.
        @return {*}
        None
        """
        super(LSTM_attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_class = n_class

        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(pretrained_weight)
        self.embedding.weight.requires_grad = update_w2v
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            bidirectional=self.bidirectional,
            dropout=drop_keep_prob,
        )

        # What is nn. Parameter(data(Tensor)=None, requires_grad=True) ? Explain
        # 理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter.
        # 所以经过类型转换这个Tensor变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。

        
        self.embedding = nn.Embedding.from_pretrained(pretrained_weight)
        self.embedding.weight.requires_grad = update_w2v
        
        # LSTM中的dropout参数保持不变
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            bidirectional=self.bidirectional,
            dropout=drop_keep_prob,  # LSTM层间的dropout
        )

        self.weight_W = nn.Parameter(torch.Tensor(2 * hidden_dim, 2 * hidden_dim))
        self.weight_proj = nn.Parameter(torch.Tensor(2 * hidden_dim, 1))

        if self.bidirectional:
            # 增加relu激活函数
            self.decoder1 = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LeakyReLU(inplace=True),
            )
            # 增加dropout层
            self.decoder2 = nn.Sequential(
                nn.Linear(hidden_dim, n_class),
            )
        else:
            self.decoder1 = nn.Linear(hidden_dim, hidden_dim)
            self.decoder2 = nn.Linear(hidden_dim, n_class)

        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

    def forward(self, inputs):
        # inputs: (num_embeddings, embedding_dim) => [batch, seq_len, embed_dim]:[32,25,50]
        # print(f"inputs shape: {inputs.shape}")
        embeddings = self.embedding(inputs)


        # states, hidden = self.encoder(embeddings.permute([0, 1, 2]))
        # 在 LSTM 的前向传播中，通常需要将输入的形状调整为 [seq_len, batch_size, embedding_dim]，以便 LSTM 能够正确处理
        # states shape: (seq_len, batch, D*hidden_size), D=2 if bidirectional = True else 1, =>[25,32,512]
        states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
        # attention:
        # u shape: (seq_len, batch, 2*D*hidden_size), D=2 if bidirectional = True else 1, =>[25,32,512]
        u = torch.tanh(torch.matmul(states, self.weight_W))
        # att shape: (seq_len, batch, 1), =>[25,32,1]
        att = torch.matmul(u, self.weight_proj)
        # att_score shape: (seq_len, batch, 1), =>[25,32,1]
        att_score = F.softmax(att, dim=1)
        # scored_x shape: (seq_len, batch, D*hidden_size), =>[25,32,512]
        scored_x = states * att_score
        # scored_x shape: (batch, seq_len, D*hidden_size), =>[32,25,512]
        scored_x = scored_x.permute([1, 0, 2])

        # encoding shape: (batch, D*hidden_size), =>[32,512]
        encoding = torch.sum(scored_x, dim=1)
        # outputs shape: (batch, n_class), =>[32,2]
        outputs = self.decoder1(encoding)
        # outputs shape: (batch, n_class), =>[32,2]
        outputs = self.decoder2(outputs)
        print(f"outputs shape: {outputs.shape}")
        print(f"att_score shape: {att_score.shape}")

        return outputs, att_score

    def visualize_attention(self, att_score, words, title):
        """
        @description: visualize the attention scores config.max_sen_len 长度
        """
        # 创建颜色映射，从白色到红色
        cmap = plt.cm.Reds
        norm = plt.Normalize(min(att_score), max(att_score))
        colors = cmap(norm(att_score))

        # 创建图形和轴
        fig, ax = plt.subplots()

        # 绘制可视化
        # 如果 words 长度大于 att_score，则截断 words
        if len(words) > len(att_score):
            words = words[:len(att_score)]
        if len(words) < len(att_score):
            # 在前面补 _PAD
            words = ["_PAD_"] * (len(att_score) - len(words)) + words    
        # 检查重复的 words 并添加索引
        word_counts = defaultdict(int)
        unique_words = []
        for word in words:
            word_counts[word] += 1
            if word_counts[word] > 1:
                # 给重复的词加上索引
                count = word_counts[word]
                unique_words.append(f"{word}_{count}")
            else:
                unique_words.append(word)
        # 打印
        print(f"words: {words}")
        print(f"att_score: {att_score}")
        # allow duplicate words
        bars = ax.barh(unique_words, att_score, color=colors)
        ax.set_xlabel('Attention Score')
        ax.set_title('Attention Scores Visualization')

        # 添加颜色条，用于显示颜色和注意力分数的映射关系
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)  # Specify the axes for the colorbar

        plt.savefig(f"./{title}.png")

import collections
import os
import random

import gensim
import torch

PADDING_TOKEN = '<PAD>'


class SentimentAnalysisDictionary(torch.nn.Module):
    """
    词典
    """

    def __init__(self, train_tokens=[]):
        super().__init__()
        if len(train_tokens) == 0:
            return

        self.dic = gensim.corpora.Dictionary(train_tokens)
        if 0 not in self.dic.token2id:  # 词典中添加'<PAD>'占用index=0，用于padding标记
            self.dic.add_documents([[PADDING_TOKEN]], prune_at=None)
            self.dic.patch_with_special_tokens({PADDING_TOKEN: 0})
        self.size = len(self.dic.token2id)
        self.word_id_dict = collections.defaultdict(lambda: -1, self.dic.token2id)

    def forward(self, x):
        return self.embedding(x)

    def save(self, filename):
        path = os.path.dirname(__file__)
        absolute_path = os.path.join(path, filename)
        torch.save({
            'dic': self.dic,
            'size': self.size,
            # 'word_id_dict': self.word_id_dict,
            'model_state_dict': self.state_dict(),
        }, absolute_path)

    @classmethod
    def load(cls, filename):
        path = os.path.dirname(__file__)
        absolute_path = os.path.join(path, filename)
        checkpoint = torch.load(f=absolute_path, weights_only=False)
        self = cls()
        self.dic = checkpoint['dic']
        self.size = checkpoint['size']
        self.word_id_dict = collections.defaultdict(lambda: random.randint(1, self.size - 1), self.dic.token2id)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.eval()
        return self

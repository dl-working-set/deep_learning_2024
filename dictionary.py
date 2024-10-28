import collections
import random

import gensim
import torch

PADDING_TOKEN = '<PAD>'


class SentimentAnalysisDictionary(torch.nn.Module):
    """
    词典
    """

    def __init__(self, train_tokens=[]):
        super(SentimentAnalysisDictionary, self).__init__()
        if len(train_tokens) == 0:
            return

        # self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        # self.embedding.weight.data.uniform_(-0.1, 0.1)

        self.dic = gensim.corpora.Dictionary(train_tokens)
        if 0 not in self.dic.token2id:  # 词典中添加'<PAD>'占用index=0，用于padding标记
            self.dic.add_documents([[PADDING_TOKEN]], prune_at=None)
            self.dic.patch_with_special_tokens({PADDING_TOKEN: 0})
        self.size = len(self.dic.token2id)
        self.word_id_dict = collections.defaultdict(lambda: random.randint(1, self.size - 1), self.dic.token2id)

    def forward(self, x):
        return self.embedding(x)


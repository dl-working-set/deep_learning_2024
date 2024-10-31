import os

import gensim
import torch


class WordEmbedding(torch.nn.Module):
    def __init__(self, tokens=[], vector_size=100, epochs=10, min_count=1):
        super(WordEmbedding, self).__init__()
        if len(tokens) == 0:
            return

        self.vector_size = vector_size
        word2vec = gensim.models.Word2Vec(sentences=tokens, vector_size=self.vector_size, sg=1,
                                          epochs=epochs, min_count=min_count)
        self.vectors = torch.tensor(word2vec.wv.vectors)
        self.key_to_index = word2vec.wv.key_to_index

    def to(self, device):
        super().to(device)
        self.vectors = self.vectors.to(device)
        return self

    def save(self, filename):
        path = os.path.dirname(__file__)
        absolute_path = os.path.join(path, filename)
        torch.save({
            'vector_size': self.vector_size,
            'vectors': self.vectors,
            'key_to_index': self.key_to_index,
            'model_state_dict': self.state_dict(),
        }, absolute_path)

    @classmethod
    def load(cls, filename):
        path = os.path.dirname(__file__)
        absolute_path = os.path.join(path, filename)
        checkpoint = torch.load(f=absolute_path, weights_only=False)
        self = cls()
        self.vector_size = checkpoint['vector_size']
        self.vectors = checkpoint['vectors']
        self.key_to_index = checkpoint['key_to_index']
        self.load_state_dict(checkpoint['model_state_dict'])
        return self

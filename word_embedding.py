import gensim
import numpy
import torch
import dataset


class WordEmbedding(torch.nn.Module):
    def __init__(self, sentences=[], stopwords=[], word2vec_vector_size=100, word2vec_epochs=5, word2vec_min_count=5):
        super(WordEmbedding, self).__init__()
        if len(sentences) == 0:
            return

        self.word2vec_vector_size = word2vec_vector_size
        # 1. 数据预处理
        self.stopwords = stopwords
        tokens = dataset.tokenization(sentences, self.stopwords)

        # 2、构建字典/词向量
        # TODO 先使用gensim构建：使用skip-gram模型训练
        self.word2vec = gensim.models.Word2Vec(sentences=tokens, vector_size=word2vec_vector_size, sg=1,
                                               epochs=word2vec_epochs, min_count=word2vec_min_count)

    def sentences2vecs(self, sentences=[]):
        tokens = dataset.tokenization(sentences, self.stopwords)
        return self.tokens2vecs(tokens)

    def tokens2vecs(self, tokens=[]):
        vecs = []
        for token in tokens:
            token_vectors = []
            for word in token:
                if word in self.word2vec.wv:
                    token_vectors.append(self.word2vec.wv[word])
                else:
                    token_vectors.append(numpy.random.uniform(-0.25, 0.25, self.word2vec_vector_size))
            vecs.append(token_vectors)
        return vecs

    def save(self, filename):
        torch.save({
            'word2vec_vector_size': self.word2vec_vector_size,
            'stopwords': self.stopwords,
            'word2vec': self.word2vec,
            'model_state_dict': self.state_dict(),
        }, filename)

    @classmethod
    def load(cls, filename):
        self = cls()
        checkpoint = torch.load(filename)
        self.word2vec_vector_size = checkpoint['word2vec_vector_size']
        self.stopwords = checkpoint['stopwords']
        self.word2vec = checkpoint['word2vec']
        self.load_state_dict(checkpoint['model_state_dict'])
        return self

    def vec_padding(self, vec, sequence_length):
        if sequence_length > len(vec):
            # 填充部分使用0值
            vec += [numpy.zeros(self.word2vec_vector_size)] * (sequence_length - len(vec))
        return vec

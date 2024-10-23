import logging
import re

import gensim
import jieba
import numpy
import torch


class WordEmbedding(torch.nn.Module):
    def __init__(self, sentences=[], stopwords=[], word2vec_vector_size=100, word2vec_epochs=5, word2vec_min_count=5):
        super(WordEmbedding, self).__init__()
        if len(sentences) == 0:
            return

        self.word2vec_vector_size = word2vec_vector_size
        # 1. 数据预处理
        self.stopwords = stopwords
        tokens = self.__sentences_preprocessing(sentences, self.stopwords)

        # 2、构建字典/词向量
        # TODO 先使用gensim构建：使用skip-gram模型训练
        self.word2vec = gensim.models.Word2Vec(sentences=tokens, vector_size=word2vec_vector_size, sg=1,
                                               epochs=word2vec_epochs, min_count=word2vec_min_count)

    def __clean_redundant(self, content):
        # TODO 去除评论中无效数据
        content = re.sub(' ', '', content)  # 清除空内容
        content = re.sub('【.*?】', '', content)
        return content

    def __sentences_preprocessing(self, sentences, stopwords):
        tokens = []
        for sentence in sentences:
            try:
                # 1. 移除无效内容：超链接、不可识别的表情等
                content = self.__clean_redundant(sentence.strip())
                # 2. 分词 [jieba]
                seg_list = jieba.cut(content, cut_all=False)
                # 3. 去除停用词
                token = []
                for seg in seg_list:
                    # 去除停用词
                    if seg not in stopwords:
                        token.append(seg)
                # if len(token) > 0:
                #     tokens.append(token)
                tokens.append(token)
            except Exception as e:
                logging.error(f'sentences preprocessing error - {e}')
        return tokens

    def sentences2vecs(self, sentences=[]):
        tokens = self.__sentences_preprocessing(sentences, self.stopwords)
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
            'stopwords ': self.stopwords,
            'word2vec ': self.word2vec,
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

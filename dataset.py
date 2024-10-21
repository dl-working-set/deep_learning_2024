"""
- 数据预处理
  - 移除无效内容：超链接、不可识别的表情等
  - 分词：jieba中文分词
  - 移除停用词：哈工大停用词
  - 构建词向量：gensim或nlp作业
  - DataLoader：训练集、验证集、测试集
"""
import collections
import datetime
import json
import logging
import re
from fileinput import filename

import gensim
import jieba
import numpy
import torch
from matplotlib import pyplot as plt


class SentimentAnalysisDataset:
    def __init__(self, stopwords_path=[], train_data_path="", validation_data_path="", test_data_path="",
                 sequence_length=100, word2vec_vector_size=100, word2vec_epochs=10, word2vec_min_count=2,
                 ):
        assert train_data_path != "" and train_data_path is not None
        assert validation_data_path != "" and validation_data_path is not None
        assert test_data_path != "" and test_data_path is not None
        assert sequence_length > 0 and sequence_length is not None
        self.stopwords_path = stopwords_path
        self.train_data_path = train_data_path
        self.validation_data_path = validation_data_path
        self.test_data_path = test_data_path
        self.sequence_length = sequence_length
        self.word2vec_vector_size = word2vec_vector_size
        self.word2vec_epochs = word2vec_epochs
        self.word2vec_min_count = word2vec_min_count

        self.emotion2label = {
            '非常满意': 0,
            '满意': 1,
            '中立': 2,
            '不满意': 3,
            '非常不满意': 4,
            '无效评论': 5
        }
        self.label2emotion = {
            0: '非常满意',
            1: '满意',
            2: '中立',
            3: '不满意',
            4: '非常不满意',
            5: '无效评论'
        }

        # 1. 数据预处理
        self.train_tokens, self.train_labels, self.validation_tokens, self.validation_labels, self.test_tokens, self.test_labels = self.__preprocessing(
            stopwords_path=self.stopwords_path, train_data_path=self.train_data_path,
            validation_data_path=self.validation_data_path, test_data_path=self.test_data_path)

        # 2、构建字典/词向量
        # TODO 先使用gensim构建：使用skip-gram模型训练
        self.word2vec = gensim.models.Word2Vec(self.__corpus(), vector_size=word2vec_vector_size, sg=1,
                                               epochs=word2vec_epochs, min_count=word2vec_min_count)

        # 3、Align Sequence
        # 3.1 Truncating：tokens截断
        train_tokens_truncated = [token[:sequence_length] for token in self.train_tokens]
        validation_tokens_truncated = [token[:sequence_length] for token in self.validation_tokens]
        test_tokens_truncated = [token[:sequence_length] for token in self.test_tokens]

        # 3.2 token2vec：转为词向量，padding
        self.train_vecs = [self.__token2vecs_and_padding(token, sequence_length) for token in train_tokens_truncated]
        self.validation_vecs = [self.__token2vecs_and_padding(token, sequence_length) for token in
                                validation_tokens_truncated]
        self.test_vecs = [self.__token2vecs_and_padding(token, sequence_length) for token in test_tokens_truncated]

        # 4、构建dataset
        self.train_dataset = torch.utils.data.TensorDataset(torch.Tensor(numpy.array(self.train_vecs)),
                                                            torch.LongTensor(self.train_labels))
        self.validation_dataset = torch.utils.data.TensorDataset(torch.Tensor(numpy.array(self.validation_vecs)),
                                                                 torch.LongTensor(self.validation_labels))
        self.test_dataset = torch.utils.data.TensorDataset(torch.Tensor(numpy.array(self.test_vecs)),
                                                           torch.LongTensor(self.test_labels))

    def __preprocessing(self, stopwords_path=[], train_data_path="", validation_data_path="", test_data_path="", ):
        # 1. 加载原始数据
        train_raw_data = load_file(train_data_path)
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Train data length: {len(train_raw_data)}')
        validation_raw_data = load_file(validation_data_path)
        print(
            f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Validation data length: {len(validation_raw_data)}')
        test_raw_data = load_file(test_data_path)
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Test data length: {len(test_raw_data)}')

        # 2. 加载停用词表
        stopwords = numpy.concatenate([load_file(filename) for filename in stopwords_path], axis=0)

        # 3. 数据预处理：分词、去除评论中无效数据、去除停用词
        train_tokens, train_labels = self.__tokenization(train_raw_data, stopwords)  # 8579条
        validation_tokens, validation_labels = self.__tokenization(validation_raw_data, stopwords)  # 1991条
        test_tokens, test_labels = self.__tokenization(test_raw_data, stopwords)  # 1991条
        return train_tokens, train_labels, validation_tokens, validation_labels, test_tokens, test_labels

    def __token2vecs_and_padding(self, token, sequence_length):
        vecs = []
        for seg in token:
            if seg in self.word2vec.wv:
                vecs.append(self.word2vec.wv[seg])
        if sequence_length > len(vecs):
            # 填充部分使用0值
            vecs += [numpy.zeros(self.word2vec.wv.vector_size)] * (sequence_length - len(vecs))
        return vecs

    def __clean_redundant(self, content):
        # TODO 去除评论中无效数据
        content = re.sub(' ', '', content)  # 清除空内容
        content = re.sub('【.*?】', '', content)
        return content

    def __tokenization(self, raw_data, stopwords):
        tokens = []
        labels = []
        for json_str in raw_data:
            try:
                json_obj = json.loads(json_str)
                content = json_obj.get('content')
                label = json_obj.get('prediction')
                # 移除收尾空格
                content = content.strip()
                # 去除评论中无效数据
                content = self.__clean_redundant(content)
                # jieba分词
                seg_list = jieba.cut(content, cut_all=False)
                token = []
                for seg in seg_list:
                    # 去除停用词
                    if seg not in stopwords:
                        token.append(seg)
                label = self.emotion2label.get(label)
                if len(token) > 0 and label is not None:
                    tokens.append(token)
                    labels.append(label)
            except Exception as e:
                logging.error(f'Tokenization error - {e}')
        return tokens, labels

    # 由于样本中标签分布不均匀，在不改变训练数据的前提下，使用“权重才采样器”均衡每个minibatch数据，提升模型拟合度
    def construct_dataloader(self, batch_size=128, ):
        labels_counts = numpy.bincount(self.train_labels)  # 样本数量
        labels_weights = 1. / labels_counts  # 标签权重
        samples_weights = labels_weights[self.train_labels]  # 样本权重
        train_sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weights,
                                                               num_samples=len(samples_weights),
                                                               replacement=True)
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=batch_size,
                                                   sampler=train_sampler)
        validation_loader = torch.utils.data.DataLoader(dataset=self.validation_dataset, batch_size=batch_size,
                                                        shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, validation_loader, test_loader,

    def __corpus(self):  # 获得语料库（仅训练集）
        return self.train_tokens


def load_raw_data(filename):
    data = []
    try:
        with open(filename, encoding="utf-8") as f:
            for json_array_line in f.readlines():  # JSON格式数据
                json_array = json.loads(json_array_line)
                for json_obj in json_array:
                    if not json_obj:
                        continue
                    else:
                        data.append(json_obj)
    except FileNotFoundError:
        logging.error("No raw_data file found.")
    return data


def load_stopwords(stopwords_path=[]):
    """
    读取停用词表
    :param stopwords_path: 停用词表路径
    :return: 停用词表
    """
    stopwords = []
    try:
        for path in stopwords_path:
            with open(path, "r") as file:
                while True:
                    line = file.readline()
                    if not line:
                        break
                    stopwords.append(line.strip('\n'))
    except FileNotFoundError:
        logging.warning("No stopwords file found.")
    return stopwords


def load_file(filename):
    data = []
    try:
        with open(filename, "r") as file:
            while True:
                line = file.readline()
                if not line:
                    break
                data.append(line.strip('\n'))
    except FileNotFoundError:
        logging.Error("File not found.")
    return data


def tokens_pie(title, tokens):
    """
    打印分词长度分布图
    :param title:
    :param tokens:
    :return:
    """
    token_len = [len(token) for token in tokens]
    token_len_counter = collections.Counter(token_len)
    token_len_sorted = sorted(token_len_counter.items(), key=lambda x: x[1], reverse=True)
    token_len, counts = zip(*token_len_sorted)
    plt.bar(x=token_len, height=counts)
    plt.title(title)
    plt.xlim(0, int(max(token_len) * 0.15))
    plt.show()


def labels_pie(title, labels, emotion2label):
    """
    打印标签分布饼图
    :param title:
    :param labels:
    :return:
    """
    labels_count = collections.Counter(labels)
    labels_names = []
    labels_values = []
    for key, value in labels_count.items():
        for emotion, label in emotion2label.items():
            if key == label:
                labels_names.append(emotion)
                labels_values.append(value)
    plt.pie(labels_values, labels=labels_names, autopct='%2.2f%%', startangle=90)
    plt.title(title)
    plt.axis('equal')
    plt.show()

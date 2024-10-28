"""
- 数据预处理
  - 移除无效内容：超链接、不可识别的表情等
  - 分词：jieba中文分词
  - 移除停用词：哈工大停用词
  - 构建词向量：gensim或nlp作业
  - DataLoader：训练集、验证集、测试集
"""
import collections
import json
import logging
import re

import jieba
import numpy
import torch
from matplotlib import pyplot as plt


class SentimentAnalysisDataset:
    def __init__(self, tokens=[], emotions=[], sequence_length=50, dictionary=None, ):
        if len(tokens) == 0:
            return

        self.tokens = tokens
        # 1. Align Sequence
        #   1.1 Truncating：向量截断
        tokens_truncated = [token[:sequence_length] for token in tokens]

        #   1.2 Padding
        tokens_padded = [padding(token, sequence_length) for token in tokens_truncated]

        #   1.3 tokens to ids
        self.tokens_ids = [[dictionary.word_id_dict[word] for word in token] for token in tokens_padded]

        #   1.4 emotions to labels
        self.labels = [emotion2label.get(emotion) for emotion in emotions]

        # 2. 构建dataset
        self.dataset = torch.utils.data.TensorDataset(torch.LongTensor(numpy.array(self.tokens_ids)),
                                                      torch.LongTensor(self.labels))

    # 由于样本中标签分布不均匀，在不改变训练数据的前提下，使用“权重才采样器”均衡每个minibatch数据，提升模型拟合度
    def construct_dataloader(self, batch_size=128, ):
        labels_counts = numpy.bincount(self.labels)  # 样本数量
        labels_weights = 1. / labels_counts  # 标签权重
        samples_weights = labels_weights[self.labels]  # 样本权重
        train_sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weights,
                                                               num_samples=len(samples_weights),
                                                               replacement=True)
        return torch.utils.data.DataLoader(dataset=self.dataset, batch_size=batch_size, sampler=train_sampler)


def padding(token, sequence_length):
    while sequence_length > len(token):
        token.append('<PAD>')
    return token


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


def load_contents_labels(filename):
    contents = []
    emotions = []
    try:
        with open(filename, "r") as file:
            while True:
                line = file.readline()
                if not line:
                    break
                json_obj = json.loads(line.strip('\n'))
                content = json_obj.get('content')
                emotion = json_obj.get('prediction')
                if len(content) > 0 and emotion in emotion2label:
                    contents.append(content)
                    emotions.append(emotion)
    except FileNotFoundError:
        logging.Error(f"File[{filename}] not found.")
    return contents, emotions


def tokenization(sentences, stopwords):
    tokens = []
    for sentence in sentences:
        try:
            # 1. 移除无效内容：超链接、不可识别的表情等
            content = clean_redundant(sentence.strip())
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


def clean_redundant(content):
    # TODO 去除评论中无效数据
    content = re.sub(' ', '', content)  # 清除空内容
    content = re.sub('【.*?】', '', content)
    return content


def tokens_pie(title, tokens):
    """
    打印分词长度分布图
    :param title:
    :param tokens:
    :return:
    """
    token_len = [len(token) for token in tokens]
    token_len_counter = collections.Counter(token_len)
    token_len_sorted = sorted(token_len_counter.items(), key=lambda x: x[0])
    token_len, counts = zip(*token_len_sorted)
    sum = 0
    for i in range(len(token_len)):
        sum += counts[i]
        print(f'token长度{token_len[i]} 占比{sum / len(tokens):.4f}%')
    plt.bar(x=token_len, height=counts)
    plt.title(title)
    plt.xlim(0, int(max(token_len) * 0.15))
    plt.show()


def labels_pie(title, labels):
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
        if key in emotion2label:
            labels_names.append(emotion2label[key])  # plt不支持中文，使用label
            labels_values.append(value)
    plt.pie(labels_values, labels=labels_names, autopct='%2.2f%%', startangle=90)
    plt.title(title)
    plt.axis('equal')
    plt.show()


emotion2label = {
    '非常满意': 0,
    '满意': 1,
    '中立': 2,
    '不满意': 3,
    '非常不满意': 4,
    '无效评论': 5
}

label2emotion = {
    0: '非常满意',
    1: '满意',
    2: '中立',
    3: '不满意',
    4: '非常不满意',
    5: '无效评论'
}

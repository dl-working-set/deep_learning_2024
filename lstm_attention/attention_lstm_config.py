# !usr/bin/env python
# -*- coding:utf-8 -*-



class Config:
    update_w2v = True  # 是否在训练中更新w2v
    vocab_size = 33462  # 词汇量，与word2id中的词汇量一致
    n_class = 6  # 分类数：分别为pos和neg
    max_sen_len = 26  # 句子最大长度
    embedding_dim = 50  # 词向量维度
    batch_size = 128  # 批处理尺寸
    hidden_dim = 128  # 隐藏层节点数
    n_epoch = 8  # 训练迭代周期，即遍历整个训练样本的次数
    lr = 0.001  # 学习率；若opt=‘adadelta'，则不需要定义学习率
    drop_keep_prob = 0.5  # dropout层，参数keep的比例
    num_layers = 2  # LSTM层数
    bidirectional = True  # 是否使用双向LSTM
    model_dir = "./model"
    stopword_path = "./data/stopwords.txt"

    train_json_path = "./data/train.ndjson"
    val_json_path = "./data/validation.ndjson"
    test_json_path = "./data/test.ndjson"

    train_path = "./data/train.csv"
    val_path = "./data/validation.csv"
    test_path = "./data/test.csv"

    pre_path = "./data/pre.txt"
    word2id_path = "./word2vec/word2id.txt"
    pre_word2vec_path = "./word2vec/wiki_word2vec_50.bin"
    corpus_word2vec_path = "./word2vec/word_vec.txt"
    model_state_dict_path = "./model/sen_model.pkl"  # 训练模型保存的地址
    best_model_path = "./model/sen_model_best.pkl"

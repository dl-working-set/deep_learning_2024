import torch

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 模型参数
MODEL_TYPE = "GRU"  # TODO 模型类型
MODEL_HIDDEN_SIZE = 512  # 模型隐藏层大小
MODEL_DROPOUT_PROBS = 0.2  # 模型Dropout比例，取值 (0, 1)
MODEL_ACTIVATION = "ReLU"  # TODO 模型激活函数

# 训练参数
SEQUENCE_LENGTH = 50  # 序列长度
WORD2VEC_VECTOR_SIZE = 100  # 词向量维度
WORD2VEC_EPOCHS = 10  # 词向量训练轮次
WORD2VEC_MIN_COUNT = 2  # 词向量最小出现次数
TRAIN_BATCH_SIZE = 128  # 训练批次大小
TRAIN_EPOCHS = 50  # 训练轮次
TRAIN_LR = 0.005  # 学习率
TRAIN_LR_SCHEDULER_FACTOR = 0.1  # 学习率衰减因子

# 数据路径配置
STOPWORDS_PATH = ["stopwords/hit_stopwords.txt", "stopwords/special_characters_stopwords.txt"]  # 停用词路径
RAW_DATA_PATH = "file/raw.ndjson"  # 原始数据路径
TRAIN_DATA_PATH = "file/train.ndjson"  # 训练集路径
VALIDATION_DATA_PATH = "file/validation.ndjson"  # 验证集路径
TEST_DATA_PATH = "file/test.ndjson"  # 测试集路径
MODEL_PTH_FILENAME = f"pt/sentiment_analysis_{MODEL_TYPE}_{MODEL_HIDDEN_SIZE}_{MODEL_DROPOUT_PROBS}_{MODEL_ACTIVATION}_{SEQUENCE_LENGTH}_{WORD2VEC_VECTOR_SIZE}_{WORD2VEC_EPOCHS}_{WORD2VEC_MIN_COUNT}_{TRAIN_BATCH_SIZE}_{TRAIN_EPOCHS}_{TRAIN_LR}_{TRAIN_LR_SCHEDULER_FACTOR}.pt"  # 模型保存路径
TRAIN_DATA_RATIO = 0.8  # 训练数据比例
VALIDATION_DATA_RATIO = 0.1  # 验证数据比例
TEST_DATA_RATIO = 0.1  # 测试数据比例

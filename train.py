"""
模型训练：
- 1. 加载训练配置
- 2. 模型训练
- 3. 模型保存
"""

import time

import torch

import config
import dataset
import dictionary
import model
import run
import word_embedding

# 1. 加载训练集数据
train_contents, train_emotions = dataset.load_contents_labels(config.TRAIN_DATA_PATH)

# 2. 加载停用词表
stopwords = dataset.load_stopwords(config.STOPWORDS_PATH)

# 3. 分词、标签化
train_tokens = dataset.tokenization(train_contents, stopwords)

# 4. 词典（效率高于word2vec）
dictionary = dictionary.SentimentAnalysisDictionary(train_tokens=train_tokens)

# 5. 构建数据集
train_dataset = dataset.SentimentAnalysisDataset(tokens=train_tokens, emotions=train_emotions,
                                                 sequence_length=config.SEQUENCE_LENGTH,
                                                 dictionary=dictionary, )

train_loader = train_dataset.construct_dataloader(config.TRAIN_BATCH_SIZE)

model = model.SentimentAnalysisModel(model_type=config.MODEL_TYPE, sequence_length=config.SEQUENCE_LENGTH,
                                     embedding_dim=config.EMBEDDING_DIM, num_embeddings=dictionary.size,
                                     hidden_size=config.MODEL_HIDDEN_SIZE, num_layers=1, output_size=6,
                                     dropout_probs=config.MODEL_DROPOUT_PROBS).to(config.DEVICE)
# 6. 加载验证集数据
validation_contents, validation_emotions = dataset.load_contents_labels(config.VALIDATION_DATA_PATH)

# 7. 分词、标签化
validation_tokens = dataset.tokenization(validation_contents, stopwords)

# 7. 构建验证集
validation_dataset = dataset.SentimentAnalysisDataset(tokens=validation_tokens, emotions=validation_emotions,
                                                      sequence_length=config.SEQUENCE_LENGTH,
                                                      dictionary=dictionary, ).dataset

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN_LR)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                          factor=config.TRAIN_LR_SCHEDULER_FACTOR,
                                                          patience=5)
for epoch in range(1, config.TRAIN_EPOCHS + 1):
    train_loss = run.train(model, train_loader, loss_fn, optimizer, config.DEVICE)
    validation_loss, validation_macro_precision, validation_macro_recall, validation_macro_f1 = run.test(model,
                                                                                                         validation_dataset,
                                                                                                         loss_fn,
                                                                                                         config.DEVICE)
    lr_scheduler.step(validation_loss)
    print('epoch[{:d}]'.format(epoch),
          'train_loss[{:f}]'.format(train_loss),
          'validation_loss[{:f}]'.format(validation_loss),
          'validation_macro_precision[{:f}]'.format(validation_macro_precision),
          'validation_macro_recall[{:f}]'.format(validation_macro_recall),
          'validation_macro_f1[{:f}]'.format(validation_macro_f1),
          'optimizer_lr[{:.2e}]'.format(optimizer.param_groups[0]['lr']),
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

# 3. 模型保存
word_embedding.save(filename=config.WORD_EMBEDDING_PTH_FILENAME)
model.save(filename=config.MODEL_PTH_FILENAME)

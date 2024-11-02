"""
模型训练：
- 1. 加载训练配置
- 2. 模型训练
- 3. 模型保存
"""
import time

import torch

import dataset
import embedding
import model
import run
from init import config

# 1. 加载训练集数据
train_contents, train_emotions = dataset.load_contents_labels(config.train_data_path)

# 2. 加载停用词表
stopwords = dataset.load_stopwords(config.stopwords_path)

# 3. 分词、标签化
train_tokens = dataset.tokenization(train_contents, stopwords)

# 4. 词向量预训练
word_embedding = embedding.WordEmbedding(tokens=(train_tokens + [[config.padding_word]]),
                                         vector_size=config.model_embedding_dim).to(config.device)

# 5. 构建数据集
train_dataset = dataset.SentimentAnalysisDataset(tokens=train_tokens, emotions=train_emotions,
                                                 sequence_length=config.model_sequence_length,
                                                 key_to_index=word_embedding.key_to_index)

train_loader = train_dataset.construct_dataloader(config.training_batch_size)

# 6. 创建模型
model = model.SentimentAnalysisModel(model_type=config.model_type, sequence_length=config.model_sequence_length,
                                     embedding_dim=config.model_embedding_dim,
                                     hidden_size=config.model_hidden_size,
                                     num_classes=config.model_num_classes,
                                     dropout_probs=config.model_dropout_probs,
                                     embedding=word_embedding).to(config.device)
# 7. 加载验证集数据
validation_contents, validation_emotions = dataset.load_contents_labels(config.validation_data_path)

# 8. 分词、标签化
validation_tokens = dataset.tokenization(validation_contents, stopwords)

# 9. 构建验证集
validation_dataset = dataset.SentimentAnalysisDataset(tokens=validation_tokens, emotions=validation_emotions,
                                                      sequence_length=config.model_sequence_length,
                                                      key_to_index=word_embedding.key_to_index, ).dataset

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.training_learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                          factor=config.training_lr_scheduler_factor,
                                                          patience=5)
early_stopping = run.EarlyStopping(patience=5)

for epoch in range(1, config.training_epochs + 1):
    train_loss = run.train(model, train_loader, loss_fn, optimizer, config.device)
    validation_loss, validation_macro_precision, validation_macro_recall, validation_macro_f1 = run.test(model,
                                                                                                         validation_dataset,
                                                                                                         loss_fn,
                                                                                                         config.device)

    print('epoch[{:d}]'.format(epoch),
          'train_loss[{:f}]'.format(train_loss),
          'validation_loss[{:f}]'.format(validation_loss),
          'validation_macro_precision[{:f}]'.format(validation_macro_precision),
          'validation_macro_recall[{:f}]'.format(validation_macro_recall),
          'validation_macro_f1[{:f}]'.format(validation_macro_f1),
          'optimizer_lr[{:.2e}]'.format(optimizer.param_groups[0]['lr']),
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # early stopping
    if early_stopping(validation_loss, model):
        break
    # 学习率更新
    lr_scheduler.step(validation_loss)

# 3. 模型保存
word_embedding.save(filename=config.embedding_pt_path)
model.save(filename=config.model_pt_path)

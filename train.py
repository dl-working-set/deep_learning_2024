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
import model
import run
import word_embedding
import matplotlib.pyplot as plt

model = model.SentimentAnalysisModel(model_type=config.MODEL_TYPE, sequence_length=config.SEQUENCE_LENGTH,
                                     input_size=config.WORD2VEC_VECTOR_SIZE, hidden_size=config.MODEL_HIDDEN_SIZE,
                                     num_layers=1, output_size=6, dropout_probs=config.MODEL_DROPOUT_PROBS).to(
    config.DEVICE)

train_contents, _ = dataset.load_contents_labels(config.TRAIN_DATA_PATH)
stopwords = dataset.load_stopwords(config.STOPWORDS_PATH)

word_embedding = word_embedding.WordEmbedding(sentences=train_contents, stopwords=stopwords,
                                              word2vec_vector_size=config.WORD2VEC_VECTOR_SIZE,
                                              word2vec_epochs=config.WORD2VEC_EPOCHS,
                                              word2vec_min_count=config.WORD2VEC_MIN_COUNT)

train_loader = dataset.SentimentAnalysisDataset(data_path=config.TRAIN_DATA_PATH, word_embedding=word_embedding,
                                                sequence_length=config.SEQUENCE_LENGTH, ).construct_dataloader(
    batch_size=config.TRAIN_BATCH_SIZE)

validation_dataset = dataset.SentimentAnalysisDataset(data_path=config.VALIDATION_DATA_PATH,
                                                     word_embedding=word_embedding,
                                                     sequence_length=config.SEQUENCE_LENGTH, ).dataset

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN_LR, weight_decay=0.01)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                          factor=config.TRAIN_LR_SCHEDULER_FACTOR,
                                                          patience=5)

train_losses = []  # 用于保存训练损失
validation_losses = []  # 用于保存验证损失

for epoch in range(1, config.TRAIN_EPOCHS + 1):
    train_loss = run.train(model, train_loader, loss_fn, optimizer, config.DEVICE)
    validation_loss, validation_macro_precision, validation_macro_recall, validation_macro_f1 = run.test(model,
                                                                                                         validation_dataset,
                                                                                                         loss_fn,
                                                                                                         config.DEVICE)
    
    # 保存损失信息
    train_losses.append(train_loss)
    validation_losses.append(validation_loss)

    lr_scheduler.step(validation_loss)
    print('epoch[{:d}]'.format(epoch),
          'train_loss[{:f}]'.format(train_loss),
          'validation_loss[{:f}]'.format(validation_loss),
          'validation_macro_precision[{:f}]'.format(validation_macro_precision),
          'validation_macro_recall[{:f}]'.format(validation_macro_recall),
          'validation_macro_f1[{:f}]'.format(validation_macro_f1),
          'optimizer_lr[{:.2e}]'.format(optimizer.param_groups[0]['lr'])
          )

# 训练指标可视化保存
plt.figure()
plt.plot(range(1, config.TRAIN_EPOCHS + 1), train_losses, label='Train Loss')
plt.plot(range(1, config.TRAIN_EPOCHS + 1), validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('training_validation_loss.png')  # 保存图像
plt.close()

# 3. 模型保存
word_embedding.save(filename=config.WORD_EMBEDDING_PTH_FILENAME)
model.save(filename=config.MODEL_PTH_FILENAME)

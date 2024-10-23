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

model = model.SentimentAnalysisModel(sequence_length=config.SEQUENCE_LENGTH, input_size=config.WORD2VEC_VECTOR_SIZE,
                                     hidden_size=config.MODEL_HIDDEN_SIZE, num_layers=1, output_size=6,
                                     dropout_probs=config.MODEL_DROPOUT_PROBS).to(config.DEVICE)

train_contents, _ = dataset.load_contents_labels(config.TRAIN_DATA_PATH)
stopwords = dataset.load_stopwords(config.STOPWORDS_PATH)

word_embedding = word_embedding.WordEmbedding(sentences=train_contents, stopwords=stopwords,
                                              word2vec_vector_size=config.WORD2VEC_VECTOR_SIZE,
                                              word2vec_epochs=config.WORD2VEC_EPOCHS,
                                              word2vec_min_count=config.WORD2VEC_MIN_COUNT)

train_loader = dataset.SentimentAnalysisDataset(data_path=config.TRAIN_DATA_PATH, word_embedding=word_embedding,
                                                sequence_length=config.SEQUENCE_LENGTH, ).construct_dataloader(
    batch_size=config.TRAIN_BATCH_SIZE)

validation_loader = dataset.SentimentAnalysisDataset(data_path=config.VALIDATION_DATA_PATH,
                                                     word_embedding=word_embedding,
                                                     sequence_length=config.SEQUENCE_LENGTH, ).construct_dataloader(
    batch_size=config.TRAIN_BATCH_SIZE)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN_LR)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                          factor=config.TRAIN_LR_SCHEDULER_FACTOR,
                                                          patience=5)
for epoch in range(1, config.TRAIN_EPOCHS + 1):
    train_loss = run.train(model, train_loader, loss_fn, optimizer, config.DEVICE)
    validation_loss, validation_macro_precision, validation_macro_recall, validation_macro_f1 = run.test(model,
                                                                                                         validation_loader,
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

"""
模型训练：
- 1. 加载训练配置
- 2. 模型训练
- 3. 模型保存
"""

import torch
import time
import model
import config
import run
import dataset

model = model.SentimentAnalysisModel(input_size=config.WORD2VEC_VECTOR_SIZE, sequence_length=config.SEQUENCE_LENGTH,
                                     hidden_size=config.MODEL_HIDDEN_SIZE, num_layers=1, output_size=6,
                                     dropout=config.MODEL_DROPOUT).to(
    config.DEVICE)

train_loader, validation_loader, test_loader = (
    dataset.SentimentAnalysisDataset(stopwords_path=config.STOPWORDS_PATH,
                                     train_data_path=config.TRAIN_DATA_PATH,
                                     validation_data_path=config.VALIDATION_DATA_PATH,
                                     test_data_path=config.TEST_DATA_PATH,
                                     sequence_length=config.SEQUENCE_LENGTH,
                                     word2vec_vector_size=config.WORD2VEC_VECTOR_SIZE,
                                     word2vec_epochs=config.WORD2VEC_EPOCHS,
                                     word2vec_min_count=config.WORD2VEC_MIN_COUNT,
                                     ).construct_dataloader(batch_size=config.DATA_LOADER_BATCH_SIZE))
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN_LR)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.TRAIN_LR_SCHEDULER_GAMMA,
                                                          patience=5, verbose=True)
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
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
# 3. 模型保存
model.save(filename=config.MODEL_PTH_FILENAME)

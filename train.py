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
                                     hidden_size=config.MODEL_HIDDEN_SIZE, num_layers=1, output_size=6).to(
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
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.TRAIN_LR_SCHEDULER_STEP_SIZE,
                                               gamma=config.TRAIN_LR_SCHEDULER_GAMMA)
t = time.strftime("%m%d%H%M")
# writer = SummaryWriter('/workspace/tensorboard')
for epoch in range(1, config.TRAIN_EPOCHS + 1):
    train_loss = run.train(model, train_loader, loss_fn, optimizer, lr_scheduler, config.DEVICE)
    test_loss, test_macro_precision, test_macro_recall, test_macro_f1 = run.test(model, test_loader, loss_fn,
                                                                                 config.DEVICE)

    print('epoch[{:d}]'.format(epoch),
          'train_loss[{:f}]'.format(train_loss),
          'test_loss[{:f}]'.format(test_loss),
          # 'train_macro_p[{:f}]'.format(train_macro_p),
          'test_macro_p[{:f}]'.format(test_macro_precision),
          # 'train_macro_r[{:f}]'.format(train_macro_r),
          'test_macro_r[{:f}]'.format(test_macro_recall),
          # 'train_macro_f1[{:f}]'.format(train_macro_f1),
          'test_macro_f1[{:f}]'.format(test_macro_f1),
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # writer.add_scalars("疫情微博情绪分类__损失_{:s}".format(t), {
    #     'train_loss': train_loss,
    #     'test_loss': test_loss
    # }, epoch)
    # writer.add_scalars("疫情微博情绪分类__宏精准率_{:s}".format(t), {
    #     'train_macro_p': train_macro_p,
    #     'test_macro_p': test_macro_p
    # }, epoch)
    # writer.add_scalars("疫情微博情绪分类__宏召回率_{:s}".format(t), {
    #     'train_macro_r': train_macro_r,
    #     'test_macro_r': test_macro_r
    # }, epoch)
    # writer.add_scalars("疫情微博情绪分类__宏F1值_{:s}".format(t), {
    #     'train_macro_f1': train_macro_f1,
    #     'test_macro_f1': test_macro_f1
    # }, epoch)

# 3. 模型保存：文件名格式：sentiment_analysis_<epochs>_<lr>_<测试集大小>_<隐藏层大小>_<层数>_<输出层大小>.pt
filename = 'pt/sentiment_analysis_{:s}_{:s}_{:s}_{:s}.pt'.format(config.TRAIN_EPOCHS, config.TRAIN_LR, )
model.save(filename=filename)

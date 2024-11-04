import logging

import numpy
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV

import dataset
import embedding
import model
import run
from init import config

# 定义需要搜索的超参数及其候选值
param_grid = {
    'hidden_size': [64, 128, 256],
    'dropout_probs': [0.2, 0.4],
    'learning_rate': [0.0001, 0.001, 0.01],
    'embedding_dim': [40, 80],
    'sequence_length': [22, 31],  # 80+%, 90+%
    'batch_size': [64, 128],
}


class SentimentAnalysisEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_size=64, dropout_probs=0.3, learning_rate=0.001, embedding_dim=60, sequence_length=35,
                 batch_size=64):
        self.hidden_size = hidden_size
        self.dropout_probs = dropout_probs
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        self.model = None
        self.word_embedding = None
        self.train_loader = None
        self.validation_dataset = None
        self.classes_ = None

    def fit(self, X_train, y_train, X_val, y_val):
        # 设置 classes_ 属性
        self.classes_ = numpy.unique([key for key in dataset.label2emotion.keys()])

        # 加载训练集数据
        train_tokens = dataset.tokenization(X_train, stopwords)

        # 词向量预训练
        self.word_embedding = init_word_embedding(self.embedding_dim)

        # 构建训练集
        train_dataset = dataset.SentimentAnalysisDataset(tokens=train_tokens, emotions=y_train,
                                                         sequence_length=self.sequence_length,
                                                         key_to_index=self.word_embedding.key_to_index)
        self.train_loader = train_dataset.construct_dataloader(self.batch_size)

        # 加载验证集数据
        validation_tokens = dataset.tokenization(X_val, stopwords)

        # 构建验证集
        self.validation_dataset = dataset.SentimentAnalysisDataset(tokens=validation_tokens, emotions=y_val,
                                                                   sequence_length=self.sequence_length,
                                                                   key_to_index=self.word_embedding.key_to_index).dataset

        # 创建模型
        self.model = model.SentimentAnalysisModel(model_type=config.model_type, sequence_length=self.sequence_length,
                                                  embedding_dim=self.embedding_dim,
                                                  hidden_size=self.hidden_size,
                                                  num_classes=config.model_num_classes,
                                                  dropout_probs=self.dropout_probs,
                                                  embedding=self.word_embedding).to(config.device)

        # 定义损失函数和优化器
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                  factor=config.training_lr_scheduler_factor,
                                                                  patience=5)
        early_stopping = run.EarlyStopping(patience=10)

        # 训练模型
        for epoch in range(1, config.training_epochs + 1):
            train_loss = run.train(self.model, self.train_loader, loss_fn, optimizer, config.device)
            validation_loss, validation_macro_precision, validation_macro_recall, validation_macro_f1 = run.test(
                self.model,
                self.validation_dataset,
                loss_fn,
                config.device)

            log_message = (
                f'Epoch {epoch} - Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}, '
                f'Validation F1: {validation_macro_f1:.4f}, LR: {optimizer.param_groups[0]["lr"]:.2e}, '
                f'Hyperparameters: {self.batch_size}|{self.dropout_probs}|{self.embedding_dim}|{self.hidden_size}|{self.learning_rate}|{self.sequence_length}|, '
                f'EarlyStopping: {early_stopping.counter}/{early_stopping.patience}')

            print(log_message)
            logging.info(log_message)

            # 早停机制
            if early_stopping(validation_loss, self.model):
                print(f'Early stopping at epoch {epoch}, stopping {early_stopping.counter} - {early_stopping.patience}')
                logging.info(
                    f'Early stopping at epoch {epoch}, stopping {early_stopping.counter} - {early_stopping.patience}')
                break

            # 学习率调度
            lr_scheduler.step(validation_loss)

        return self

    def predict(self, X):
        # 加载k折验证数据
        test_tokens = dataset.tokenization(X, stopwords)
        test_dataset = dataset.SentimentAnalysisDataset(tokens=test_tokens, emotions=['中立'] * len(test_tokens),
                                                        sequence_length=self.sequence_length,
                                                        key_to_index=self.word_embedding.key_to_index, ).dataset

        self.model.eval()
        data = test_dataset.tensors[0].to(config.device)
        output = self.model(data)
        predicted = output.argmax(dim=1).cpu().numpy()
        return [dataset.label2emotion.get(p) for p in predicted]


# 初始化词向量
def init_word_embedding(embedding_dim):
    train_contents, train_emotions = dataset.load_contents_labels(config.train_data_path)
    stopwords = dataset.load_stopwords(config.stopwords_path)
    # validation_contents, validation_emotions = dataset.load_contents_labels(config.validation_data_path)
    train_tokens = dataset.tokenization(train_contents, stopwords)
    # validation_tokens = dataset.tokenization(validation_contents, stopwords)
    word_embedding = embedding.WordEmbedding(tokens=(train_tokens + [[config.padding_word]]),
                                             vector_size=embedding_dim).to(config.device)
    return word_embedding


# 加载训练集数据
train_contents, train_emotions = dataset.load_contents_labels(config.train_data_path)
# 加载停用词表
stopwords = dataset.load_stopwords(config.stopwords_path)
# 加载验证集数据
validation_contents, validation_emotions = dataset.load_contents_labels(config.validation_data_path)
# 创建模型估计器
estimator = SentimentAnalysisEstimator()

# 网格搜索
grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=make_scorer(f1_score, average='macro'),
                           cv=3, n_jobs=2, refit=True, verbose=2)
grid_search.fit(X=train_contents, y=train_emotions, X_val=validation_contents, y_val=validation_emotions)

# 输出最佳参数和最佳得分
print(f'Best F1 Score: {grid_search.best_score_}')
print(f'Best Parameters: {grid_search.best_params_}')
logging.info(f'Best F1 Score: {grid_search.best_score_}')
logging.info(f'Best Parameters: {grid_search.best_params_}')

# 测试集验证
test_contents, test_emotions = dataset.load_contents_labels(config.test_data_path)

# 使用最佳模型进行预测
best_estimator = grid_search.best_estimator_
predictions = best_estimator.predict(test_contents)
# 打印预测结果
for content, prediction in zip(test_contents, predictions):
    print(f'Text: {content}, Prediction: {prediction}')

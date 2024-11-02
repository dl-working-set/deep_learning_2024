"""
  - 模型训练：默认超参数，损失趋势
  - 模型验证集验证：指标、趋势
"""
import torch
from sklearn.metrics import precision_score, recall_score, f1_score


def train(model, train_loader, loss_fn, optimizer, device):
    """训练函数

    :param model:训练模型
    :param train_loader:训练集
    :param loss_fn:损失函数
    :param optimizer:优化器
    :param device: 设备
    :return:平均损失
    """
    model.train()
    running_loss = 0.0
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        output = model(data)
        loss = loss_fn(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def test(model, test_dataset, loss_fn, device):
    """测试函数

    :param model:测试模型
    :param test_dataset:测试集
    :param loss_fn:损失函数
    :param device: 设备
    :return:平均损失，准确率，召回率，F1值
    """
    model.eval()
    with torch.no_grad():
        data, targets = test_dataset.tensors[0].to(device), test_dataset.tensors[1].to(device)
        output = model(data)
        loss = loss_fn(output, targets)
        running_loss = loss.item()

        predicted = output.argmax(dim=1).cpu().numpy()
        labels = targets.cpu().numpy()
        macro_precision = precision_score(labels, predicted, average='macro', zero_division=0)  # 宏精准率
        macro_recall = recall_score(labels, predicted, average='macro', zero_division=0)  # 宏召回率
        macro_f1 = f1_score(labels, predicted, average='macro')  # 宏F1值
        return running_loss, macro_precision, macro_recall, macro_f1


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        早停类

        :param patience: 允许验证损失不下降的次数
        :param verbose: 是否打印早停信息
        :param delta: 最小变化阈值，验证损失必须至少减少 delta 才算作改进
        :param path: 模型保存路径
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0  # 重置计数器

        if self.verbose:
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

        return self.early_stop

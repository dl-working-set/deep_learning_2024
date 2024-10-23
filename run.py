"""
  - 模型训练：默认超参数，损失趋势
  - 模型验证集验证：指标、趋势
"""
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


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


def test(model, test_loader, loss_fn, device):
    """测试函数

    :param model:测试模型
    :param test_loader:测试集
    :param loss_fn:损失函数
    :param device: 设备
    :return:平均损失，准确率，召回率，F1值
    """
    model.eval()
    running_loss = 0.0
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            loss = loss_fn(output, targets)
            running_loss += loss.item()

            predicted = output.argmax(dim=1).cpu().numpy()
            labels = targets.cpu().numpy()
            macro_precision += precision_score(labels, predicted, average='macro', zero_division=0)  # 宏精准率
            macro_recall += recall_score(labels, predicted, average='macro', zero_division=0)  # 宏召回率
            macro_f1 += f1_score(labels, predicted, average='macro')  # 宏F1值
    return running_loss / len(test_loader), macro_precision / len(test_loader), macro_recall / len(
        test_loader), macro_f1 / len(test_loader)

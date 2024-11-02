"""
统计训练数据中各个标签的数量
"""
import random
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import Counter
from init import config

TRAIN_DATA_PATH, VALIDATION_DATA_PATH, TEST_DATA_PATH, RAW_DATA_PATH = config.train_data_path, config.validation_data_path, config.test_data_path, config.raw_data_path

# 设置 matplotlib 中文显示

mpl.rcParams['font.sans-serif'] = ['Heiti TC']
mpl.rcParams['axes.unicode_minus'] = False  

prediction_types = ["非常满意", "满意", "中立", "不满意", "非常不满意", "无效评论"]
prediction_map = {t: list() for t in prediction_types}



# 读取原始数据，对数据按比例进行划分
train_data_percentage = 0.8
validation_data_percentage = 0.1
test_data_percentage = 0.1

train_data = []
validation_data = []
test_data = []


# 读取原始数据文件，并按比例划分训练、验证、测试数据，各个数据文件中的标签比例保持一致
with open(RAW_DATA_PATH, "r") as f:
    for line in f:
        if line:
            data = json.loads(line)
            if data["prediction"] in prediction_types:
                prediction_map[data["prediction"]].append(data)

for prediction in prediction_types:
    data_len = len(prediction_map[prediction])
    train_end = int(data_len * train_data_percentage)
    validation_end = int(data_len * (validation_data_percentage + train_data_percentage))

    train_data.extend(prediction_map[prediction][:train_end])
    validation_data.extend(prediction_map[prediction][train_end:validation_end])
    test_data.extend(prediction_map[prediction][validation_end:])

print(f"训练数据中各个标签的数量为：{Counter([data['prediction'] for data in train_data])}")
print(f"验证数据中各个标签的数量为：{Counter([data['prediction'] for data in validation_data])}")
print(f"测试数据中各个标签的数量为：{Counter([data['prediction'] for data in test_data])}")

# 打乱数据
random.shuffle(train_data)
random.shuffle(validation_data)
random.shuffle(test_data)

# 使用 plt 绘制各个数据集中各个标签的数量
# 绘制 训练数据中各个标签的数量, train_data, 绘制为饼图
plt.figure(figsize=(8, 6))
train_data_counter = Counter([data['prediction'] for data in train_data])
plt.pie([train_data_counter[t] for t in prediction_types], labels=[f"{t} ({train_data_counter[t]})" for t in prediction_types], autopct='%1.1f%%')
plt.title(f"训练数据中各个标签的数量({len(train_data)})")
plt.show()

# 绘制 验证数据中各个标签的数量, validation_data, 绘制为饼图
validation_data_counter = Counter([data['prediction'] for data in validation_data])
plt.pie([validation_data_counter[t] for t in prediction_types], labels=[f"{t} ({validation_data_counter[t]})" for t in prediction_types], autopct='%1.1f%%')
plt.title(f"验证数据中各个标签的数量({len(validation_data)})")
plt.show()

# 绘制 测试数据中各个标签的数量, test_data, 绘制为饼图
test_data_counter = Counter([data['prediction'] for data in test_data])
plt.pie([test_data_counter[t] for t in prediction_types], labels=[f"{t} ({test_data_counter[t]})" for t in prediction_types], autopct='%1.1f%%')
plt.title(f"测试数据中各个标签的数量({len(test_data)})")
plt.show()

# 保存训练、验证、测试数据
with open(TRAIN_DATA_PATH, "w", encoding="utf-8") as f:
    for data in train_data:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")

with open(VALIDATION_DATA_PATH, "w", encoding="utf-8") as f:
    for data in validation_data:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")

with open(TEST_DATA_PATH, "w", encoding="utf-8") as f:
    for data in test_data:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")

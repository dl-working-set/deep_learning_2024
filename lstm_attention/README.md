# 基于LSTM+Attention的情感分析
## 参考内容
- [基于LSTM+Attention的情感分析](https://github.com/SoulDGXu/Sentiment-Analysis-Chinese-pytorch)
- [基于Attention-Bi-LSTM的微博评论情感分析研究](https://pdf.hanspub.org/csa20201200000_48123814.pdf)

## 代码结构
- attention_lstm_config.py: 配置文件
- attention_lstm_data_process.py: 数据处理
- attention_lstm_model.py: 模型定义
- attention_lstm_train.py: 模型训练
- attention_lstm_eval.py: 模型评估
- data/: 数据集和停用词表
- model/: 模型文件
- attention_png: 注意力机制可视化图片
- word2vec: 词向量相关文件

## 运行过程
python 3.10+
1. 安装依赖
    ```bash 
    pip install -r requirements.txt
    ```
2. 执行 attention_lstm_train.py 进行训练



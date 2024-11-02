import logging
import os

import torch
import yaml


class Config:
    def __init__(self, config_data):
        self.model = config_data.get('model', {})
        self.training = config_data.get('training', {})
        self.path = config_data.get('path', {})

        # 替换路径中的占位符
        self._replace_placeholders()

    def _replace_placeholders(self):
        """替换路径中的占位符"""
        self.path['model_pt'] = self.path.get('model_pt').format(
            model_type=self.model.get('type'),
            model_hidden_size=self.model.get('hidden_size'),
            model_dropout_probs=self.model.get('dropout_probs'),
            model_activation=self.model.get('activation'),
            model_embedding_dim=self.model.get('embedding_dim'),
            model_sequence_length=self.model.get('sequence_length'),
            training_batch_size=self.training.get('batch_size'),
            training_epochs=self.training.get('epochs'),
            training_learning_rate=self.training.get('learning_rate'),
            training_lr_scheduler_factor=self.training.get('lr_scheduler_factor')
        )

    @property
    def model_type(self):
        return self.model.get('type')

    @property
    def model_hidden_size(self):
        return self.model.get('hidden_size')

    @property
    def model_dropout_probs(self):
        return self.model.get('dropout_probs')

    @property
    def model_activation(self):
        return self.model.get('activation')

    @property
    def model_embedding_dim(self):
        return self.model.get('embedding_dim')

    @property
    def model_sequence_length(self):
        return self.model.get('sequence_length')

    @property
    def model_num_classes(self):
        return self.model.get('num_classes')

    @property
    def training_batch_size(self):
        return self.training.get('batch_size')

    @property
    def training_epochs(self):
        return self.training.get('epochs')

    @property
    def training_learning_rate(self):
        return self.training.get('learning_rate')

    @property
    def training_lr_scheduler_factor(self):
        return self.training.get('lr_scheduler_factor')

    @property
    def stopwords_path(self):
        return self.path.get('stopwords', [])

    @property
    def raw_data_path(self):
        return self.path.get('raw_data')

    @property
    def train_data_path(self):
        return self.path.get('train_data')

    @property
    def validation_data_path(self):
        return self.path.get('validation_data')

    @property
    def test_data_path(self):
        return self.path.get('test_data')

    @property
    def model_pt_path(self):
        return self.path.get('model_pt')

    @property
    def embedding_pt_path(self):
        return self.path.get('embedding_pt')

    @property
    def device(self):
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    @property
    def padding_word(self):
        return '<PAD>'


# 获取当前文件的绝对路径
path = os.path.dirname(__file__)
config_path = os.path.join(path, 'config.yaml')

# 加载配置文件
with open(config_path, 'r') as stream:
    try:
        config_data = yaml.safe_load(stream)
        # print(config_data)
    except yaml.YAMLError as exc:
        logging.error(f'Error loading config.yaml - {exc}')
        exit(1)

path = os.path.dirname(__file__)
absolute_path = os.path.join(path, 'pt')
if not os.path.exists(absolute_path):
    os.makedirs(absolute_path)

config = Config(config_data)
__all__ = ['config']

logging.basicConfig(level=logging.INFO)

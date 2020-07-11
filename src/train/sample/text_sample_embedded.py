from typing import List
import torch
from learning_framework.src.train.sample.base_sample import BaseSample
import numpy as np


class TextSampleEmbedded(BaseSample):
    padding_size = 400

    def __init__(self, text_data: np.ndarray, text_gt: np.ndarray):
        assert isinstance(text_data, np.ndarray), "{} is not supported, only str".format(type(text_data))
        assert isinstance(text_gt, np.ndarray), "{} is not supported, only str".format(type(text_gt))

        super().__init__({'text_data': text_data, 'text_gt': text_gt})

    @property
    def text_data(self):
        return self['text_data']

    @text_data.setter
    def text_data(self, value):
        self['text_data'] = value

    @property
    def text_gt(self):
        return self['text_gt']

    @text_gt.setter
    def text_gt(self, value):
        self['text_gt'] = value

    @staticmethod
    def _pad(data: np.ndarray, padding_size: int):
        if padding_size < data.shape[0]:
            print("warning: data size larger than padding. Data size: {}, padding: {}".format(data.shape[0], padding_size))
            final_size = padding_size
        else:
            final_size = data.shape[0]

        data_padded = np.zeros((padding_size, data.shape[1]))
        data_padded[0:final_size, :] = data[0:final_size, :]
        return data_padded

    def add_padding(self):
        self.text_data = self._pad(self.text_data, self.padding_size)
        self.text_gt = self._pad(self.text_gt, self.padding_size)

    def convert_to_torch(self):
        self.text_data = torch.from_numpy(self.text_data).float()
        self.text_gt = torch.from_numpy(self.text_gt).float()

    def __iter__(self):
        return iter((self.text_data, self.text_gt))

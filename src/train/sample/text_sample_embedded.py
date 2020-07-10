from typing import List
import torch
from learning_framework.src.train.sample.i_sample import ISample
import numpy as np


class TextSampleEmbedded(ISample):
    def __init__(self, text: np.ndarray):
        assert isinstance(text, np.ndarray), "{} is not supported, only str".format(type(text))
        self.text = text

    def convert_to_torch(self):
        self.text = torch.from_numpy(self.text)

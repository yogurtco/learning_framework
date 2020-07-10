from typing import List
import torch
from learning_framework.src.train.sample.i_sample import ISample


class TextSample(ISample):
    def __init__(self, text: List[str]):
        assert isinstance(text, List), "{} is not supported, only str".format(type(text))
        self.text = text

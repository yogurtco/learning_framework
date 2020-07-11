from typing import List
import torch
from learning_framework.src.train.sample.base_sample import BaseSample


class TextSample(BaseSample):
    def __init__(self, text: List[str]):
        assert isinstance(text, List), "{} is not supported, only str".format(type(text))
        super().__init__({'text': text})

    @property
    def text(self):
        return self['text']

from typing import List
import numpy as np

from learning_framework.src.train.preprocess.i_preprocess import IPreprocess
from learning_framework.src.train.sample.i_sample import ISample
from learning_framework.src.train.sample.text_sample import TextSample
from learning_framework.src.train.sample.text_sample_embedded import TextSampleEmbedded


class CharacterEmbeddingPreprocess(IPreprocess):
    max_ascii_num = 255

    def __call__(self, sample: ISample):
        assert isinstance(sample, TextSample), "Type {} is not supported".format(type(sample))

        combined_text = "\n".join(sample.text)
        embedded_text = np.zeros((len(combined_text), self.max_ascii_num))
        for index, c in enumerate(combined_text):
            embedded_text[index, ord(c)] = 1

        return TextSampleEmbedded(embedded_text)

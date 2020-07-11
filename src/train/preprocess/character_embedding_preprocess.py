from typing import List
import numpy as np

from learning_framework.src.train.preprocess.i_preprocess import IPreprocess
from learning_framework.src.train.sample.base_sample import BaseSample
from learning_framework.src.train.sample.text_sample import TextSample
from learning_framework.src.train.sample.text_sample_embedded import TextSampleEmbedded


class CharacterEmbeddingPreprocess(IPreprocess):
    max_ascii_num = 255 + 1 # plus 1 for un

    def __call__(self, sample: BaseSample):
        assert isinstance(sample, TextSample), "Type {} is not supported".format(type(sample))

        combined_text = "\n".join(sample.text)
        embedded_text = np.zeros((len(combined_text), self.max_ascii_num))
        for index, c in enumerate(combined_text):

            char_index = ord(c)
            if ord(c) >= self.max_ascii_num:
                print("{} is not supported".format(c))
                char_index = -1
            embedded_text[index, char_index] = 1

        return TextSampleEmbedded(embedded_text[0:-1, :], embedded_text[1::, :])

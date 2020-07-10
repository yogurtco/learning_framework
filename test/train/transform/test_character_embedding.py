from learning_framework.src.train.sample.text_sample import TextSample
from learning_framework.src.train.preprocess.character_embedding_preprocess import CharacterEmbeddingPreprocess
import numpy as np


def test_character_embedding():
    text_to_embed = ['The cat and the bat, sat on a hat', '!@#$%^&*()', '12343098', '"`', '"''_-+=/.,\\''']
    embedding = CharacterEmbeddingPreprocess()(TextSample(text_to_embed)).text

    for i in range(embedding.shape[0]):
        assert np.sum(embedding[i, :]) == 1

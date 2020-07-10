from abc import ABC, abstractmethod

from torchvision.transforms import Compose

from learning_framework.src.train.preprocess.i_preprocess import IPreprocess


class ComposePreprocess(Compose, IPreprocess):
    """
    Transforms are performed after data loading
    """
    pass

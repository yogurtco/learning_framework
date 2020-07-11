from abc import ABC, abstractmethod

from learning_framework.src.train.sample.base_sample import BaseSample


class IPreprocess(ABC):
    """
    Transforms are performed after data loading
    """

    @abstractmethod
    def __call__(self, sample: BaseSample):
        """
        Perform preprocess
        :return: transformed object
        """
        pass

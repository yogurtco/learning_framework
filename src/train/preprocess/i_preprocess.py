from abc import ABC, abstractmethod

from learning_framework.src.train.sample.i_sample import ISample


class IPreprocess(ABC):
    """
    Transforms are performed after data loading
    """

    @abstractmethod
    def __call__(self, sample: ISample):
        """
        Perform preprocess
        :return: transformed object
        """
        pass

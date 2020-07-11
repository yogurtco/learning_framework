from abc import ABC, abstractmethod


class BaseSample(ABC, dict):
    """
    Represents a sampled data
    """
    def __init__(self, data):
        super().__init__(data)

    @abstractmethod
    def convert_to_torch(self):
        """
        Convert relevant inner fields to torch
        """
        pass
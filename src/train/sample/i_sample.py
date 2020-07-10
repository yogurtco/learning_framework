from abc import ABC, abstractmethod


class ISample(ABC):
    """
    Represents a sampled data
    """
    def convert_to_torch(self):
        """
        Convert relevant inner fields to torch
        """
        pass
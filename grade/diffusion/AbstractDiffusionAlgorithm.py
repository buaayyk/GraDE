from abc import ABC, abstractmethod

from torchmetrics.wrappers import abstract

class DiffusionAlgorithm(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train_step(self, data):
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, data):
        raise NotImplementedError

    @abstractmethod
    def test_step(self, data):
        raise NotImplementedError

    @abstractmethod
    def generate_step(self):
        raise NotImplementedError

    @abstractmethod
    def estimate_step(self, data):
        raise NotImplementedError

    @abstractmethod
    def get_model(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_optimizer(self):
        raise NotImplementedError

    @abstractmethod
    def get_scheduler(self):
        raise NotImplementedError

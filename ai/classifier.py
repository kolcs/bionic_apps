from abc import ABC, abstractmethod
from enum import Enum, auto


class ClassifierType(Enum):
    SVM = auto()
    VGG19 = auto()
    DENSENET201 = auto()


class ClassifierInterface(ABC):

    @abstractmethod
    def fit(self, x, y, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError

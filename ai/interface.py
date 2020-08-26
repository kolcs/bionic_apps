from abc import ABC, abstractmethod


class ClassifierInterface(ABC):

    @abstractmethod
    def fit(self, x, y, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError

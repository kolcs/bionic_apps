from abc import ABC, abstractmethod

from tensorflow import keras


class BaseNetwork(ABC):

    def __init__(self, input_shape):
        self._input_shape = input_shape

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class SimpleFeedForwardNetwork(BaseNetwork):

    def __init__(self, input_shape):
        super().__init__(input_shape)
        self._model = None

    def build_model(self):
        x = keras.Input(shape=self._input_shape, )

    def fit(self):
        pass

    def predict(self):
        pass


if __name__ == '__main__':
    nn = BaseNetwork(0)

from abc import ABC, abstractmethod

from numpy import argmax as np_argmax
from tensorflow import keras


class ClassifierInterface(ABC):

    @abstractmethod
    def fit(self, x, y, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError


class BaseNet(ClassifierInterface):

    def __init__(self, input_shape, output_shape):
        self._input_shape = input_shape
        self._output_shape = output_shape

        input_tensor, outputs = self._build_graph()
        self._create_model(input_tensor, outputs)

    def _create_model(self, input_tensor, outputs):
        self._model = keras.Model(inputs=input_tensor, outputs=outputs)
        self._model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

    def _build_graph(self):
        raise NotImplementedError("Model build function is not implemented")

    def predict(self, x):
        predictions = self._model.predict(x)
        return np_argmax(predictions, axis=-1)

    def fit(self, x, y=None, *, validation_data=None, batch_size=None, epochs=8):
        self._model.fit(x, y, validation_data=validation_data, batch_size=batch_size, epochs=epochs)

    def summary(self):
        self._model.summary()

    def evaluate(self, x, y=None):
        self._model.evaluate(x, y)

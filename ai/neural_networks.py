from abc import ABC, abstractmethod

import tensorflow as tf
from numpy import argmax as np_argmax
from tensorflow import keras

from ai.classifier import ClassifierInterface


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


class VGG19(ClassifierInterface):

    def __init__(self, classes, input_shape, weights="imagenet"):
        input_tensor = keras.layers.Input(shape=input_shape)
        x = input_tensor

        if len(input_shape) == 2:
            x = keras.layers.Lambda(lambda tens: tf.expand_dims(tens, axis=-1))(x)
        if len(input_shape) == 2 or len(input_shape) == 3 and input_shape[2] == 1:
            x = keras.layers.Lambda(lambda tens: tf.image.grayscale_to_rgb(tens))(x)

        base_model = keras.applications.VGG19(
            include_top=False,
            weights=weights,
            input_tensor=x,
            input_shape=None,
            pooling=None,
        )

        # add end node
        x = keras.layers.Flatten(name='flatten')(base_model.outputs[0])
        x = keras.layers.Dense(4096, activation='relu', name='fc1')(x)
        x = keras.layers.Dense(4096, activation='relu', name='fc2')(x)
        x = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
        # create new model
        self._model = keras.Model(input_tensor, x, name='bci-vgg19')
        self._model.compile(
            optimizer=keras.optimizers.Adam(),
            # preprocessing.LabelEncoder() is required
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

    def predict(self, x):
        predictions = self._model.predict(x)
        return np_argmax(predictions, axis=-1)

    def fit(self, x, y, **kwargs):
        self._model.fit(x, y, batch_size=32)


if __name__ == '__main__':
    nn = VGG19(2, (64, 251))

from abc import ABC, abstractmethod

import tensorflow as tf
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

    def __init__(self, classes, input_shape):
        input_tensor = keras.layers.Input(shape=input_shape)
        x = input_tensor

        if len(input_shape) == 2:
            x = keras.layers.Lambda(lambda tens: tf.expand_dims(tens, axis=-1))(input_tensor)
            x = keras.layers.Lambda(lambda tens: tf.image.grayscale_to_rgb(tens))(x)

        base_model = keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
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
        self._model.summary()
        self._model.compile(
            optimizer=keras.optimizers.Adam(),
            # preprocessing.LabelEncoder() is required
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

    def predict(self, x):
        pass

    def fit(self, x, y, **kwargs):
        pass


if __name__ == '__main__':
    nn = VGG19(2, (64, 251))

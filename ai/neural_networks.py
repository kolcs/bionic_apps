from abc import ABC, abstractmethod

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
        base_model = keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
        )
        # change input layer
        base_model.summary()
        base_model.layers.pop(0)
        base_model.summary()
        input_layer = keras.layers.Input(shape=input_shape)

        # todo: expand to gray + expand dims --> 2d pic to rgb pic
        # tf.image.grayscale_to_rgb(images, name=None)

        x = keras.layers.Conv2D(3, (3, 3),
                                activation='relu',
                                padding='same',
                                name='pre_conv')(input_layer)
        outputs = base_model(x)
        # add end node
        x = keras.layers.Flatten(name='flatten')(outputs[0])
        x = keras.layers.Dense(4096, activation='relu', name='fc1')(x)
        x = keras.layers.Dense(4096, activation='relu', name='fc2')(x)
        x = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
        # create new model
        self._model = keras.Model(input_layer, x, name='bci-vgg19')
        self._model.summary()

    def predict(self, x):
        pass

    def fit(self, x, y, **kwargs):
        pass


if __name__ == '__main__':
    nn = VGG19(2, (64, 251, 2))

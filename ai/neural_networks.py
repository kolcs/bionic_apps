from enum import Enum, auto

import tensorflow as tf
from numpy import argmax as np_argmax
from tensorflow import keras

from ai.interface import ClassifierInterface


class VggType(Enum):
    VGG16 = auto()
    VGG19 = auto()


class VGG(ClassifierInterface):

    def __init__(self, net_type, classes, input_shape, weights="imagenet"):
        # tf.compat.v1.reset_default_graph()
        input_tensor = keras.layers.Input(shape=input_shape)
        x = input_tensor

        if len(input_shape) == 2:
            x = keras.layers.Lambda(lambda tens: tf.expand_dims(tens, axis=-1))(x)
        if len(input_shape) == 2 or len(input_shape) == 3 and input_shape[2] == 1:
            x = keras.layers.Lambda(lambda tens: tf.image.grayscale_to_rgb(tens))(x)

        model_kwargs = dict(
            include_top=False,
            weights=weights,
            input_tensor=x,
            input_shape=None,
            pooling=None,
        )
        if net_type == VggType.VGG16:
            base_model = keras.applications.VGG16(**model_kwargs)
        elif net_type == VggType.VGG19:
            base_model = keras.applications.VGG19(**model_kwargs)
        else:
            raise NotImplementedError('VGG net {} is not defined'.format(net_type))

        # add end node
        x = keras.layers.Flatten(name='flatten')(base_model.outputs[0])
        x = keras.layers.Dense(4096, activation='relu', name='fc1')(x)
        x = keras.layers.Dense(4096, activation='relu', name='fc2')(x)
        x = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
        # create new model
        self._model = keras.Model(input_tensor, x, name='bci-vgg')
        self._model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

    def predict(self, x):
        predictions = self._model.predict(x)
        return np_argmax(predictions, axis=-1)

    def fit(self, x, y, **kwargs):
        self._model.fit(x, y, batch_size=32, epochs=10)

    def summary(self):
        self._model.summary()


class DenseNetType(Enum):
    DN121 = auto()
    DN169 = auto()
    DN201 = auto()


class DenseNet(ClassifierInterface):

    def __init__(self, net_type, classes, input_shape, weights="imagenet"):
        # tf.compat.v1.reset_default_graph()
        input_tensor = keras.layers.Input(shape=input_shape)
        x = input_tensor

        if len(input_shape) == 2:
            x = keras.layers.Lambda(lambda tens: tf.expand_dims(tens, axis=-1))(x)
        if len(input_shape) == 2 or len(input_shape) == 3 and input_shape[2] == 1:
            x = keras.layers.Lambda(lambda tens: tf.image.grayscale_to_rgb(tens))(x)

        model_kwargs = dict(
            include_top=False,
            weights=weights,
            input_tensor=x,
            input_shape=None,
            pooling='avg',
        )
        if net_type == DenseNetType.DN121:
            base_model = keras.applications.DenseNet121(**model_kwargs)
        elif net_type == DenseNetType.DN169:
            base_model = keras.applications.DenseNet169(**model_kwargs)
        elif net_type == DenseNetType.DN201:
            base_model = keras.applications.DenseNet201(**model_kwargs)
        else:
            raise NotImplementedError('DenseNet {} is not defined'.format(net_type))

        # add end node
        x = keras.layers.Dense(classes, activation='softmax', name='predictions')(base_model.outputs[0])
        # create new model
        self._model = keras.Model(input_tensor, x, name='bci-dense')
        self._model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

    def predict(self, x):
        predictions = self._model.predict(x)
        return np_argmax(predictions, axis=-1)

    def fit(self, x, y, **kwargs):
        self._model.fit(x, y, batch_size=32, epochs=8)

    def summary(self):
        self._model.summary()


class CascadeConvRecNet(ClassifierInterface):  # https://github.com/Kearlay/research/blob/master/py/eeg_main.py

    def __init__(self, classes, input_shape, conv_2d_filters=None, lstm_layers=2):
        if conv_2d_filters is None:
            conv_2d_filters = [32, 64, 128]

        input_tensor = keras.layers.Input(shape=input_shape)
        x = input_tensor
        x = keras.layers.BatchNormalization(name='batch_norm')(x)

        # Convolutional Block
        for i, filter_num in enumerate(conv_2d_filters):
            x = keras.layers.TimeDistributed(
                keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), padding='same',
                                    activation=tf.nn.leaky_relu),
                name='conv{}'.format(i)
            )(x)
        x = keras.layers.TimeDistributed(keras.layers.Flatten(), name='flatten')(x)

        # Fully connected
        x = keras.layers.TimeDistributed(
            keras.layers.Dense(units=1024, activation=tf.nn.leaky_relu),
            name='FC1'
        )(x)

        # LSTM block
        for i in range(lstm_layers):
            ret_seq = i < lstm_layers - 1
            x = keras.layers.LSTM(input_shape[0], return_sequences=ret_seq, name='LSTM{}'.format(i))(x)

        # Fully connected layer block
        x = keras.layers.Dense(1024, activation=tf.nn.leaky_relu, name='FC2')(x)
        # x = keras.layers.Dropout(0.5, name='dropout2')(x)

        # Output layer
        outputs = keras.layers.Dense(classes, activation='softmax')(x)

        # Model compile
        self._model = keras.Model(inputs=input_tensor, outputs=outputs)
        self._model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

    def predict(self, x):
        predictions = self._model.predict(x)
        return np_argmax(predictions, axis=-1)

    def fit(self, x, y, **kwargs):
        batch_size = 32
        for i in range(0, len(x), batch_size):
            self._model.fit(x[i:i + batch_size], y[i:i + batch_size], batch_size=batch_size, epochs=8)

    def summary(self):
        self._model.summary()


if __name__ == '__main__':
    nn = CascadeConvRecNet(2, (251, 64, 64, 1))
    nn.summary()

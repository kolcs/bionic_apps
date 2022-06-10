from enum import Enum, auto

import tensorflow as tf
from tensorflow import keras

from ..ai.interface import TFBaseNet


class VggType(Enum):
    VGG16 = auto()
    VGG19 = auto()


class VGG(TFBaseNet):

    def __init__(self, net_type, input_shape, classes, weights="imagenet"):
        self._net_type = net_type
        self._weights = weights
        super(VGG, self).__init__(input_shape, classes)
        # tf.compat.v1.reset_default_graph()

    def _build_graph(self):
        input_tensor = keras.layers.Input(shape=self._input_shape)
        x = input_tensor

        if len(self._input_shape) == 2:
            x = keras.layers.Lambda(lambda tens: tf.expand_dims(tens, axis=-1))(x)
        if len(self._input_shape) == 2 or len(self._input_shape) == 3 and self._input_shape[2] == 1:
            x = keras.layers.Lambda(lambda tens: tf.image.grayscale_to_rgb(tens))(x)

        model_kwargs = dict(
            include_top=False,
            weights=self._weights,
            input_tensor=x,
            input_shape=None,
            pooling=None,
        )
        if self._net_type == VggType.VGG16:
            base_model = keras.applications.VGG16(**model_kwargs)
        elif self._net_type == VggType.VGG19:
            base_model = keras.applications.VGG19(**model_kwargs)
        else:
            raise NotImplementedError('VGG net {} is not defined'.format(self._net_type))

        # add end node
        x = keras.layers.Flatten(name='flatten')(base_model.outputs[0])
        x = keras.layers.Dense(4096, activation='relu', name='fc1')(x)
        x = keras.layers.Dense(4096, activation='relu', name='fc2')(x)
        x = keras.layers.Dense(self._output_shape, activation='softmax', name='predictions')(x)
        return input_tensor, x


class DenseNetType(Enum):
    DN121 = auto()
    DN169 = auto()
    DN201 = auto()


class DenseNet(TFBaseNet):

    def __init__(self, net_type, input_shape, classes, weights="imagenet"):
        # tf.compat.v1.reset_default_graph()
        self._net_type = net_type
        self._weights = weights
        super(DenseNet, self).__init__(input_shape, classes)

    def _build_graph(self):
        input_tensor = keras.layers.Input(shape=self._input_shape)
        x = input_tensor

        if len(self._input_shape) == 2:
            x = keras.layers.Lambda(lambda tens: tf.expand_dims(tens, axis=-1))(x)
        if len(self._input_shape) == 2 or len(self._input_shape) == 3 and self._input_shape[2] == 1:
            x = keras.layers.Lambda(lambda tens: tf.image.grayscale_to_rgb(tens))(x)

        model_kwargs = dict(
            include_top=False,
            weights=self._weights,
            input_tensor=x,
            input_shape=None,
            pooling='avg',
        )
        if self._net_type == DenseNetType.DN121:
            base_model = keras.applications.DenseNet121(**model_kwargs)
        elif self._net_type == DenseNetType.DN169:
            base_model = keras.applications.DenseNet169(**model_kwargs)
        elif self._net_type == DenseNetType.DN201:
            base_model = keras.applications.DenseNet201(**model_kwargs)
        else:
            raise NotImplementedError('DenseNet {} is not defined'.format(self._net_type))

        # add end node
        x = keras.layers.Dense(self._output_shape, activation='softmax', name='predictions')(base_model.outputs[0])
        return input_tensor, x


class CascadeConvRecNet(TFBaseNet):  # https://github.com/Kearlay/research/blob/master/py/eeg_main.py

    def __init__(self, input_shape, classes, conv_2d_filters=None, lstm_layers=2):
        if conv_2d_filters is None:
            conv_2d_filters = [32, 64, 128]
        self._conv_2d_filters = conv_2d_filters
        self._lstm_layers = lstm_layers
        super(CascadeConvRecNet, self).__init__(input_shape, classes)

    def _build_graph(self):
        input_tensor = keras.layers.Input(shape=self._input_shape)
        x = input_tensor
        x = keras.layers.BatchNormalization(name='batch_norm')(x)

        # Convolutional Block
        for i, filter_num in enumerate(self._conv_2d_filters):
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
        x = keras.layers.TimeDistributed(
            keras.layers.Dropout(0.5),
            name='dropout1'
        )(x)

        # LSTM block
        for i in range(self._lstm_layers):
            ret_seq = i < self._lstm_layers - 1
            x = keras.layers.LSTM(self._input_shape[0], return_sequences=ret_seq, name='LSTM{}'.format(i))(x)

        # Fully connected layer block
        x = keras.layers.Dense(1024, activation=tf.nn.leaky_relu, name='FC2')(x)
        x = keras.layers.Dropout(0.5, name='dropout2')(x)

        # Output layer
        outputs = keras.layers.Dense(self._output_shape, activation='softmax')(x)
        return input_tensor, outputs


class BasicNet(TFBaseNet):

    def __init__(self, input_shape, classes, resize_shape=(64, 64), resize_method='bilinear'):
        self._resize_shape = resize_shape
        self._resize_method = resize_method
        super(BasicNet, self).__init__(input_shape, classes)

    def _build_graph(self):
        input_tensor = keras.layers.Input(shape=self._input_shape)
        x = input_tensor
        if len(self._input_shape) == 2:
            x = keras.layers.Lambda(lambda tens: tf.expand_dims(tens, axis=-1))(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Lambda(lambda tens: tf.image.resize(tens, self._resize_shape, self._resize_method))(x)

        conv_filters = [16, 32, 64]
        kernel_sizes = [(3, 3), (5, 5)]
        x_list = list()
        for kernel in kernel_sizes:
            y = x
            for filt in conv_filters:
                y = keras.layers.Conv2D(filt, kernel, activation=keras.layers.LeakyReLU())(y)
            y = keras.layers.Flatten()(y)
            x_list.append(y)
        x = keras.layers.concatenate(x_list)

        x = keras.layers.Dense(units=1024, activation=tf.nn.leaky_relu)(x)
        x = keras.layers.Dense(units=512, activation=tf.nn.leaky_relu)(x)
        x = keras.layers.Dense(units=256, activation=tf.nn.leaky_relu)(x)
        x = keras.layers.Dense(self._output_shape, activation='softmax')(x)
        return input_tensor, x


if __name__ == '__main__':
    nn = BasicNet((64, 120), 2)
    nn.summary()

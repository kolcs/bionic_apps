import tensorflow as tf
from tensorflow import keras

from bionic_apps.ai.interface import TFBaseNet


#  https://github.com/vlawhern/arl-eegmodels
class EEGNet(TFBaseNet):

    def __init__(self, input_shape, classes, dropout_rate=0.5, kernel_length=.5, f1=8,
                 d=2, f2=16, norm_rate=0.25, dropout_type='Dropout', save_path='tf_log/'):
        self.dropout_rate = dropout_rate
        if dropout_type == 'SpatialDropout2D':
            dropout_type = keras.layers.SpatialDropout2D
        elif dropout_type == 'Dropout':
            dropout_type = keras.layers.Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')
        self.dropout_type = dropout_type
        if isinstance(kernel_length, float) and kernel_length < 1:
            self.kernel_length = int(input_shape[-1] * kernel_length)
        else:
            self.kernel_length = kernel_length
        self.f1 = f1
        self.d = d
        self.f2 = f2
        self.norm_rate = norm_rate
        super(EEGNet, self).__init__(input_shape, classes, save_path)

    def _build_graph(self):
        input_tensor = keras.layers.Input(shape=self._input_shape)
        x = input_tensor

        channels = self._input_shape[0]
        samples = self._input_shape[1]

        if len(self._input_shape) == 2:
            x = keras.layers.Lambda(lambda tens: tf.expand_dims(tens, axis=-1))(x)

        block1 = keras.layers.Conv2D(self.f1, (1, self.kernel_length), padding='same',
                                     input_shape=(channels, samples, 1),
                                     use_bias=False)(x)
        block1 = keras.layers.BatchNormalization()(block1)
        block1 = keras.layers.DepthwiseConv2D((channels, 1), use_bias=False,
                                              depth_multiplier=self.d,
                                              depthwise_constraint=keras.constraints.max_norm(1.))(block1)
        block1 = keras.layers.BatchNormalization()(block1)
        block1 = keras.layers.Activation('elu')(block1)
        block1 = keras.layers.AveragePooling2D((1, 4))(block1)
        block1 = self.dropout_type(self.dropout_rate)(block1)

        block2 = keras.layers.SeparableConv2D(self.f2, (1, 16),
                                              use_bias=False, padding='same')(block1)
        block2 = keras.layers.BatchNormalization()(block2)
        block2 = keras.layers.Activation('elu')(block2)
        block2 = keras.layers.AveragePooling2D((1, 8))(block2)
        block2 = self.dropout_type(self.dropout_rate)(block2)

        flatten = keras.layers.Flatten(name='flatten')(block2)

        dense = keras.layers.Dense(self._output_shape, name='dense',
                                   kernel_constraint=keras.constraints.max_norm(self.norm_rate))(flatten)
        softmax = keras.layers.Activation('softmax', name='softmax')(dense)

        return input_tensor, softmax


#  https://github.com/vlawhern/arl-eegmodels
class DeepConvNet(TFBaseNet):

    def __init__(self, input_shape, classes, dropout_rate=0.5,
                 pool_size=(1, 2), strides=(1, 2), kernel_size=(1, 5),
                 conv_filters=(25, 50, 100, 200),
                 save_path='tf_log/'):
        """Keras implementation of the Deep Convolutional Network as described in
        Schirrmeister et. al. (2017), Human Brain Mapping.

        This implementation assumes the input is a 2-second EEG signal sampled at
        128Hz, as opposed to signals sampled at 250Hz as described in the original
        paper. We also perform temporal convolutions of length (1, 5) as opposed
        to (1, 10) due to this sampling rate difference.

        Note that we use the max_norm constraint on all convolutional layers, as
        well as the classification layer. We also change the defaults for the
        BatchNormalization layer. We used this based on a personal communication
        with the original authors.

                          ours        original paper
        pool_size        1, 2        1, 3
        strides          1, 2        1, 3
        kernel_size      1, 5        1, 10

        Note that this implementation has not been verified by the original
        authors.
        """
        self.dropout_rate = dropout_rate
        self.pool_size = pool_size
        self.strides = strides
        self.kernel_size = kernel_size
        self.conv_filters = conv_filters
        super(DeepConvNet, self).__init__(input_shape, classes, save_path)

    def _build_graph(self):
        input_tensor = keras.layers.Input(shape=self._input_shape)
        x = input_tensor

        channels = self._input_shape[0]
        samples = self._input_shape[1]

        if len(self._input_shape) == 2:
            x = keras.layers.Lambda(lambda tens: tf.expand_dims(tens, axis=-1))(x)

        x = keras.layers.Conv2D(self.conv_filters[0], self.kernel_size,
                                input_shape=(channels, samples, 1),
                                kernel_constraint=keras.constraints.max_norm(2., axis=(0, 1, 2)))(x)

        for i, filters in enumerate(self.conv_filters):
            # print(i, filters)
            kernel_size = (channels, 1) if i == 0 else self.kernel_size

            x = keras.layers.Conv2D(filters, kernel_size,
                                    kernel_constraint=keras.constraints.max_norm(2., axis=(0, 1, 2)))(x)
            x = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
            x = keras.layers.Activation('elu')(x)
            x = keras.layers.MaxPooling2D(pool_size=self.pool_size, strides=self.strides)(x)
            x = keras.layers.Dropout(self.dropout_rate)(x)

        flatten = keras.layers.Flatten()(x)

        dense = keras.layers.Dense(self._output_shape, name='dense',
                                   kernel_constraint=keras.constraints.max_norm(.5))(flatten)
        softmax = keras.layers.Activation('softmax', name='softmax')(dense)

        return input_tensor, softmax


#  https://github.com/vlawhern/arl-eegmodels
class ShallowConvNet(TFBaseNet):

    def __init__(self, input_shape, classes, dropout_rate=0.5,
                 pool_size=(1, 35), strides=(1, 7), kernel_size=(1, 13),
                 conv_filters=40,
                 save_path='tf_log/'):
        """Keras implementation of the Shallow Convolutional Network as described
        in Schirrmeister et. al. (2017), Human Brain Mapping.

        Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in
        the original paper, they do temporal convolutions of length 25 for EEG
        data sampled at 250Hz. We instead use length 13 since the sampling rate is
        roughly half of the 250Hz which the paper used. The pool_size and stride
        in later layers is also approximately half of what is used in the paper.

        Note that we use the max_norm constraint on all convolutional layers, as
        well as the classification layer. We also change the defaults for the
        BatchNormalization layer. We used this based on a personal communication
        with the original authors.

                         ours        original paper
        pool_size        1, 35       1, 75
        strides          1, 7        1, 15
        conv filters     1, 13       1, 25

        Note that this implementation has not been verified by the original
        authors. We do note that this implementation reproduces the results in the
        original paper with minor deviations.
        """
        self.dropout_rate = dropout_rate
        self.pool_size = pool_size
        self.strides = strides
        self.kernel_size = kernel_size
        self.conv_filters = conv_filters
        super(ShallowConvNet, self).__init__(input_shape, classes, save_path)

    @staticmethod
    def _log(x):
        return tf.math.log(tf.clip_by_value(x, clip_value_min=1e-7, clip_value_max=10000))

    def _build_graph(self):
        input_tensor = keras.layers.Input(shape=self._input_shape)
        x = input_tensor

        channels = self._input_shape[0]
        samples = self._input_shape[1]

        if len(self._input_shape) == 2:
            x = keras.layers.Lambda(lambda tens: tf.expand_dims(tens, axis=-1))(x)

        x = keras.layers.Conv2D(self.conv_filters, self.kernel_size,
                                input_shape=(channels, samples, 1),
                                kernel_constraint=keras.constraints.max_norm(2., axis=(0, 1, 2)))(x)
        x = keras.layers.Conv2D(self.conv_filters, (channels, 1), use_bias=False,
                                kernel_constraint=keras.constraints.max_norm(2., axis=(0, 1, 2)))(x)
        x = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
        x = keras.layers.Activation(tf.square)(x)
        x = keras.layers.MaxPooling2D(pool_size=self.pool_size, strides=self.strides)(x)
        x = keras.layers.Activation(self._log)(x)
        x = keras.layers.Dropout(self.dropout_rate)(x)

        flatten = keras.layers.Flatten()(x)

        dense = keras.layers.Dense(self._output_shape, name='dense',
                                   kernel_constraint=keras.constraints.max_norm(.5))(flatten)
        softmax = keras.layers.Activation('softmax', name='softmax')(dense)

        return input_tensor, softmax


# https://github.com/rootskar/EEGMotorImagery/blob/master/EEGModels.py
class EEGNetFusion(TFBaseNet):

    def __init__(self, input_shape, classes, dropout_rate=0.5, kernel_length=.5,
                 f1_1=8, f1_2=16, f1_3=32,
                 f2_1=16, f2_2=32, f2_3=64,
                 d1=2, d2=2, d3=2,
                 norm_rate=0.25, dropout_type='Dropout', save_path='tf_log/'):
        self.dropout_rate = dropout_rate
        if dropout_type == 'SpatialDropout2D':
            dropout_type = keras.layers.SpatialDropout2D
        elif dropout_type == 'Dropout':
            dropout_type = keras.layers.Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')
        self.dropout_type = dropout_type
        if isinstance(kernel_length, float) and kernel_length < 1:
            self.kernel_length = int(input_shape[-1] * kernel_length)
        else:
            self.kernel_length = kernel_length
        self.f11 = f1_1
        self.f12 = f1_2
        self.f13 = f1_3
        self.f21 = f2_1
        self.f22 = f2_2
        self.f23 = f2_3
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.norm_rate = norm_rate
        super(EEGNetFusion, self).__init__(input_shape, classes, save_path)

    def _build_graph(self):
        input_tensor = keras.layers.Input(shape=self._input_shape)
        x = input_tensor

        channels = self._input_shape[0]
        samples = self._input_shape[1]

        if len(self._input_shape) == 2:
            x = keras.layers.Lambda(lambda tens: tf.expand_dims(tens, axis=-1))(x)

        branch1 = keras.layers.Conv2D(self.f11, (1, self.kernel_length), padding='same',
                                      input_shape=(channels, samples, 1),
                                      use_bias=False)(x)
        branch1 = keras.layers.BatchNormalization()(branch1)
        branch1 = keras.layers.DepthwiseConv2D((channels, 1), use_bias=False,
                                               depth_multiplier=self.d1,
                                               depthwise_constraint=keras.constraints.max_norm(1.))(branch1)
        branch1 = keras.layers.BatchNormalization()(branch1)
        branch1 = keras.layers.Activation('elu')(branch1)
        branch1 = keras.layers.AveragePooling2D((1, 4))(branch1)
        branch1 = self.dropout_type(self.dropout_rate)(branch1)

        branch1 = keras.layers.SeparableConv2D(self.f21, (1, 16),
                                               use_bias=False, padding='same')(branch1)
        branch1 = keras.layers.BatchNormalization()(branch1)
        branch1 = keras.layers.Activation('elu')(branch1)
        branch1 = keras.layers.AveragePooling2D((1, 8))(branch1)
        branch1 = self.dropout_type(self.dropout_rate)(branch1)

        flatten = keras.layers.Flatten(name='flatten')(branch1)

        dense = keras.layers.Dense(self._output_shape, name='dense',
                                   kernel_constraint=keras.constraints.max_norm(self.norm_rate))(flatten)
        softmax = keras.layers.Activation('softmax', name='softmax')(dense)

        return input_tensor, softmax


if __name__ == '__main__':
    nn = ShallowConvNet((63, 128), 2)
    nn.summary()
    # keras.utils.plot_model(nn, "model.png", show_shapes=True)

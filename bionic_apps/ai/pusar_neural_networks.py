from enum import Enum, auto

import tensorflow as tf
from tensorflow import keras

from .interface import TFBaseNet


# CNNType = Enum('CNNType', 'CNN QNet EEGNet')

class CNNType(Enum):
    CNN = auto()
    QNet = auto()
    EEGNet = auto()
    PCNN = auto()


class CNN(TFBaseNet):

    def __init__(self, net_type, input_shape, classes, weights="imagenet"):
        self._net_type = net_type
        self._weights = weights
        self.classes = classes
        super(CNN, self).__init__(input_shape, classes)

    def _build_graph(self):

        input_tensor = keras.layers.Input(shape=self._input_shape)
        x = input_tensor

        if len(self._input_shape) == 2:
            x = keras.layers.Lambda(lambda tens: tf.expand_dims(tens, axis=-1))(x)
        if len(self._input_shape) == 2 or len(self._input_shape) == 3 and self._input_shape[2] == 1:
            x = keras.layers.Lambda(lambda tens: tf.image.grayscale_to_rgb(tens))(x)

        x = keras.layers.Conv2D(filters=32, activation='relu', kernel_size=3)(x)
        # add end node
        x = keras.layers.Flatten(name='flatten')(x)
        x = keras.layers.Dense(100, activation='relu', name='fc1')(x)
        x = keras.layers.Dense(self._output_shape, activation='softmax', name='predictions')(x)
        return input_tensor, x


#  https://github.com/vlawhern/arl-eegmodels
class EEGNet(TFBaseNet):

    def __init__(self, input_shape, classes, dropoutRate=0.5, kernLength=.5, F1=8,
                 D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        self.dropout_rate = dropoutRate
        if dropoutType == 'SpatialDropout2D':
            dropoutType = tf.keras.layers.SpatialDropout2D
        elif dropoutType == 'Dropout':
            dropoutType = tf.keras.layers.Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')
        self.dropout_type = dropoutType
        if isinstance(kernLength, float) and kernLength < 1:
            self.kernel_length = int(input_shape[-1] * kernLength)
        else:
            self.kernel_length = kernLength
        self.f1 = F1
        self.d = D
        self.f2 = F2
        self.norm_rate = norm_rate
        super(EEGNet, self).__init__(input_shape, classes)

    def _build_graph(self):
        input_tensor = keras.layers.Input(shape=self._input_shape)
        x = input_tensor

        channels = self._input_shape[0]
        samples = self._input_shape[1]

        if len(self._input_shape) == 2:
            x = keras.layers.Lambda(lambda tens: tf.expand_dims(tens, axis=-1))(x)

        block1 = tf.keras.layers.Conv2D(self.f1, (1, self.kernel_length), padding='same',
                                        input_shape=(channels, samples, 1),
                                        use_bias=False)(x)
        block1 = tf.keras.layers.BatchNormalization()(block1)
        block1 = tf.keras.layers.DepthwiseConv2D((channels, 1), use_bias=False,
                                                 depth_multiplier=self.d,
                                                 depthwise_constraint=tf.keras.constraints.max_norm(1.))(block1)
        block1 = tf.keras.layers.BatchNormalization()(block1)
        block1 = tf.keras.layers.Activation('elu')(block1)
        block1 = tf.keras.layers.AveragePooling2D((1, 4))(block1)
        block1 = self.dropout_type(self.dropout_rate)(block1)

        block2 = tf.keras.layers.SeparableConv2D(self.f2, (1, 16),
                                                 use_bias=False, padding='same')(block1)
        block2 = tf.keras.layers.BatchNormalization()(block2)
        block2 = tf.keras.layers.Activation('elu')(block2)
        block2 = tf.keras.layers.AveragePooling2D((1, 8))(block2)
        block2 = self.dropout_type(self.dropout_rate)(block2)

        flatten = tf.keras.layers.Flatten(name='flatten')(block2)

        dense = tf.keras.layers.Dense(self._output_shape, name='dense',
                                      kernel_constraint=tf.keras.constraints.max_norm(self.norm_rate))(flatten)
        softmax = tf.keras.layers.Activation('softmax', name='softmax')(dense)

        return input_tensor, softmax


class QNet(TFBaseNet):

    def __init__(self, net_type, input_shape, classes, weights="imagenet"):
        self._net_type = net_type
        self._weights = weights
        super(QNet, self).__init__(input_shape, classes)

    @property
    def layers(self):
        return self._model.layers

    @property
    def _is_graph_network(self):
        return self._model._is_graph_network

    @property
    def _network_nodes(self):
        return self._model._network_nodes

    def _attention_line(self, x):
        x = keras.layers.GlobalAveragePooling2D()(x)  # gap
        x = keras.layers.Lambda(lambda tens: tf.expand_dims(tens, axis=-1))(x)
        x = keras.layers.Lambda(lambda tens: tf.expand_dims(tens, axis=-1))(x)
        x = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same')(x)
        x = keras.layers.Softmax()(x)
        return x

    def expand_permute(self, ntc, tcn, cnt):
        expanded_tcn = tcn
        permuted_ntc = keras.layers.Lambda(lambda tens: tf.transpose(tens, perm=[0, 2, 3, 1]))(ntc)
        permuted_cnt = keras.layers.Lambda(lambda tens: tf.transpose(tens, perm=[0, 2, 1, 3]))(cnt)
        return permuted_ntc, expanded_tcn, permuted_cnt

    def threed_attention_module(self, input_data):
        ntc = input_data  # C-attention
        tcn = keras.layers.Lambda(lambda tens: tf.transpose(tens, perm=[0, 2, 3, 1]))(input_data)  # N-attention
        cnt = keras.layers.Lambda(lambda tens: tf.transpose(tens, perm=[0, 3, 1, 2]))(input_data)  # T-attention
        ntc = self._attention_line(ntc)
        tcn = self._attention_line(tcn)
        cnt = self._attention_line(cnt)
        ntc, tcn, cnt = self.expand_permute(ntc, tcn, cnt)
        hadamard_prod = keras.layers.Multiply()([ntc, tcn, cnt])  # shape=(None, 64, 480, 3)
        inp_conv = keras.layers.Conv2D(filters=hadamard_prod.shape[-1], kernel_size=(1, 1), padding='same')(input_data)
        el_wise_mult = keras.layers.Multiply()([inp_conv, hadamard_prod])  # shape=(None, 64, 480, 3)
        batch_norm = keras.layers.BatchNormalization()(el_wise_mult)
        relu = keras.layers.Activation('relu')(batch_norm)
        output = keras.layers.Add()([input_data, relu])
        return output

    def pre_res_unit(self, residual_input_data, filters):
        identity_x = residual_input_data
        shortcut = identity_x

        model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet"
        )

        conv1 = keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(
            residual_input_data)

        resblock = None

        if filters == 64:
            resblock = keras.Sequential(model.layers[8:13])(conv1)
        elif filters == 128:
            resblock = keras.Sequential(model.layers[40:45])(conv1)
        elif filters == 256:
            resblock = keras.Sequential(model.layers[82:87])(conv1)
        elif filters == 512:
            resblock = keras.Sequential(model.layers[144:149])(conv1)

        if residual_input_data.shape[-1] != filters:
            shortcut = keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(
                identity_x)
            shortcut = keras.layers.BatchNormalization()(shortcut)

        output = keras.layers.Add()([shortcut, resblock])
        output = keras.layers.Activation('relu')(output)
        output = keras.layers.BatchNormalization()(output)

        return output

    def _residual_unit(self, residual_input_data, filters):
        identity_x = residual_input_data
        shortcut = identity_x

        conv_op_1 = keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(
            residual_input_data)
        activ_1 = keras.layers.Activation('relu')(conv_op_1)
        batch_norm_op_1 = keras.layers.BatchNormalization()(activ_1)

        conv_op_2 = keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(
            batch_norm_op_1)
        activ_2 = keras.layers.Activation('relu')(conv_op_2)
        batch_norm_op_2 = keras.layers.BatchNormalization()(activ_2)

        if residual_input_data.shape[-1] != filters:
            shortcut = keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(
                identity_x)
            shortcut = keras.layers.BatchNormalization()(shortcut)

        output = keras.layers.Add()([shortcut, batch_norm_op_2])
        output = keras.layers.Activation('relu')(output)
        output = keras.layers.BatchNormalization()(output)

        return output

    def _branch(self, input_data, filters1, filters2):
        # res_unit_1 = self._residual_unit(input_data, filters1)
        res_unit_1 = self.pre_res_unit(input_data, filters1)
        dropout_1 = tf.keras.layers.Dropout(.5)(res_unit_1)
        threed_am_1 = self.threed_attention_module(dropout_1)
        # res_unit_2 = self._residual_unit(threed_am_1, filters2)
        res_unit_2 = self.pre_res_unit(threed_am_1, filters2)
        dropout_2 = tf.keras.layers.Dropout(.5)(res_unit_2)
        return dropout_2

    def create_bil_vector(self, br_A, br_B):
        N = int(br_A.shape[1])
        T = int(br_A.shape[2])
        z_out_1 = keras.layers.Lambda(lambda x: tf.einsum('bnti,bntj->bij', x[0], x[1]))((br_A, br_B))
        z_out_2 = keras.layers.Lambda(lambda x: tf.divide(x, N * T))(z_out_1)
        a = keras.layers.Lambda(lambda x: tf.sqrt(tf.abs(x) + 1e-12))(z_out_2)
        b = keras.layers.Lambda(lambda x: tf.sign(x))(z_out_2)
        ab = keras.layers.Multiply()([a, b])
        z_out = keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x))(ab)
        return z_out

    def _build_graph(self):
        input_tensor = keras.layers.Input(shape=self._input_shape)
        input_data = input_tensor
        if len(self._input_shape) == 2:
            input_data = keras.layers.Lambda(lambda tens: tf.expand_dims(tens, axis=-1))(
                input_data)  # shape=(None, 64, 480, 1)

        conv_layer_1 = keras.layers.Conv2D(filters=3, kernel_size=(1, 1), padding='same')(input_data)
        threed_am_1 = self.threed_attention_module(conv_layer_1)  # shape=(None, 64, 480, 3),
        conv_layer_2 = keras.layers.Conv2D(filters=64, kernel_size=(7, 7), padding='same')(
            threed_am_1)  # shape=(None, 64, 480, 64)
        dropout_1 = tf.keras.layers.Dropout(.5)(conv_layer_2)
        max_pool_layer_1 = keras.layers.MaxPool2D(pool_size=(2, 2))(dropout_1)  # shape=(None, 32, 240, 64)
        dropout_2 = tf.keras.layers.Dropout(.5)(max_pool_layer_1)

        pre_res_unit_1 = self.pre_res_unit(dropout_2, 64)
        pre_res_unit_2 = self.pre_res_unit(pre_res_unit_1, 128)

        # res_unit_1 = self._residual_unit(dropout_2, 64)  # shape=(None, 32, 240, 64)
        # res_unit_2 = self._residual_unit(res_unit_1, 128)  # shape=(None, 32, 240, 128)

        dropout_3 = tf.keras.layers.Dropout(.5)(pre_res_unit_2)
        br_A = self._branch(dropout_3, 256, 512)  # shape=(None, 32, 240, 512)
        br_B = self._branch(dropout_3, 256, 512)  # shape=(None, 32, 240, 512)

        bil_layer = self.create_bil_vector(br_A, br_B)  # shape=(None, 512, 512)

        # add end node
        flatten_layer = keras.layers.Flatten()(bil_layer)  # shape=(None, 262144)
        fully_connected_layer_1 = keras.layers.Dense(100, activation='relu')(flatten_layer)  # shape=(None, 100)
        fully_connected_layer_last = keras.layers.Dense(self._output_shape, activation='softmax')(
            fully_connected_layer_1)
        return input_tensor, fully_connected_layer_last

    def _create_model(self, input_tensor, outputs):
        self._model = keras.Model(inputs=input_tensor, outputs=outputs)

        self._model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=tf.keras.optimizers.schedules.CosineDecay(
                    1.19 * 1e-3, 1000, alpha=4.18 * 1e-9, name=None
                )
            ),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )


class PCNN(TFBaseNet):

    def __init__(self, net_type, input_shape, classes, weights=None):
        self._net_type = net_type
        self._weights = weights
        super(PCNN, self).__init__(input_shape, classes)

    def _conv_through_time(self, input_data):
        conv_1 = keras.layers.Conv2D(filters=16, kernel_size=(1, 3), padding='same', activation='selu')(input_data)
        max_pool_1 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv_1)
        batch_1 = keras.layers.BatchNormalization()(max_pool_1)
        dropout_1 = keras.layers.Dropout(.32)(batch_1)

        conv_2 = keras.layers.Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='selu')(dropout_1)
        max_pool_2 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv_2)
        batch_2 = keras.layers.BatchNormalization()(max_pool_2)
        dropout_2 = keras.layers.Dropout(.32)(batch_2)

        conv_3 = keras.layers.Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='selu')(dropout_2)
        max_pool_3 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv_3)
        batch_3 = keras.layers.BatchNormalization()(max_pool_3)
        dropout_3 = keras.layers.Dropout(.32)(batch_3)

        conv_4 = keras.layers.Conv2D(filters=128, kernel_size=(1, 3), padding='same', activation='selu')(dropout_3)
        max_pool_4 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv_4)
        batch_4 = keras.layers.BatchNormalization()(max_pool_4)
        dropout_4 = keras.layers.Dropout(.32)(batch_4)

        flat = keras.layers.Flatten()(dropout_4)

        return flat

    def _conv_through_fr(self, input_data):
        conv_1 = keras.layers.Conv2D(filters=128, kernel_size=(input_data.shape[1], 1), activation='selu')(input_data)
        batch_1 = keras.layers.BatchNormalization()(conv_1)
        dropout_1 = keras.layers.Dropout(.32)(batch_1)

        flat = keras.layers.Flatten()(dropout_1)

        return flat

    def _conv_through_time_fr(self, input_data):
        conv_1 = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='selu')(input_data)
        max_pool_1 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv_1)
        batch_1 = keras.layers.BatchNormalization()(max_pool_1)
        dropout_1 = keras.layers.Dropout(.32)(batch_1)

        conv_2 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='selu')(dropout_1)
        max_pool_2 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv_2)
        batch_2 = keras.layers.BatchNormalization()(max_pool_2)
        dropout_2 = keras.layers.Dropout(.32)(batch_2)

        conv_3 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='selu')(dropout_2)
        max_pool_3 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv_3)
        batch_3 = keras.layers.BatchNormalization()(max_pool_3)
        dropout_3 = keras.layers.Dropout(.32)(batch_3)

        conv_4 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='selu')(dropout_3)
        max_pool_4 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv_4)
        batch_4 = keras.layers.BatchNormalization()(max_pool_4)
        dropout_4 = keras.layers.Dropout(.32)(batch_4)

        flat = keras.layers.Flatten()(dropout_4)

        return flat

    def _build_graph(self):
        input_tensor = keras.layers.Input(shape=self._input_shape)
        x = input_tensor

        if len(self._input_shape) == 2:
            x = keras.layers.Lambda(lambda tens: tf.expand_dims(tens, axis=-1))(x)
        if len(self._input_shape) == 2 or len(self._input_shape) == 3 and self._input_shape[2] == 1:
            x = keras.layers.Lambda(lambda tens: tf.image.grayscale_to_rgb(tens))(x)

        # convolution through time
        conv_through_time = self._conv_through_time(x)
        # convolution through frequency
        conv_through_fr = self._conv_through_fr(x)
        # convolution through both
        conv_through_time_fr = self._conv_through_time_fr(x)
        # concatenating
        concat = keras.layers.concatenate([conv_through_time, conv_through_fr, conv_through_time_fr])

        # add end node
        fully_connected_layer_1 = keras.layers.Dense(1024, activation='relu')(concat)
        fully_connected_layer_2 = keras.layers.Dense(512, activation='relu')(fully_connected_layer_1)
        output = keras.layers.Dense(self._output_shape, activation='softmax')(fully_connected_layer_2)

        return input_tensor, output


if __name__ == '__main__':
    nn = EEGNet((64, 128), 2)
    nn.summary()
    # keras.utils.plot_model(nn, "model.png", show_shapes=True)

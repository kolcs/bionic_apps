import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import *


class SpatialTransformLayer(keras.layers.Layer):

    def __init__(self, output_dim, **kwargs):
        """
        Constructor for spatial transformation layer.
        Inner layers defined here.

        :param output_dim: output dimension after transformation
        :param kwargs:
        """
        self.output_dim = output_dim  # 3D
        super(SpatialTransformLayer, self).__init__(**kwargs)

        # batch, channel, timepoint
        self._split_data_by_timepoints = keras.layers.Lambda(
            lambda x: [x[:, :, i] for i in range(keras.backend.int_shape(x)[-1])])

        def _transform(tens):
            batch, channel = tens.shape
            assert channel == 64, "Spatial transformation was designed for 64 channels. Got {} instead".format(channel)
            y = np.zeros((batch, 10 * 11))
            for i, ind in enumerate(Physionet.CHANNEL_TRANSFORMATION):
                y[:, ind] = tens[:, i]
            return y

        self._spatial_transformation = keras.layers.Lambda(
            lambda x: tf.py_function(func=_transform, inp=[x], Tout=tf.float32))
        self._reshape = keras.layers.Reshape((self.output_dim[0], self.output_dim[1]))
        self._expand_dim = keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x))
        self._concat = tf.keras.layers.Concatenate()

    def build(self, input_shape):
        # Make sure to call the `build` method at the end
        super(SpatialTransformLayer, self).build(input_shape)

    def call(self, x):
        """
        Building graph here as functional model

        :param x: input tensor
        :return: spatial transformed model
        """
        # split
        x_list = self._split_data_by_timepoints(x)

        x_2d_list = list()
        for tensor in x_list:
            # spatial transform in 1D
            transf_x = self._spatial_transformation(tensor)
            # reshape to 2D
            transf_2d = self._reshape(transf_x)
            # extend to be able to create 3D
            x_2d_list.append(self._expand_dim(transf_2d))

        # create 3D tensor
        return self._concat(x_2d_list)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(self.output_dim).as_list()
        out_shape = tf.TensorShape(input_shape).as_list()[0] + shape
        return tf.TensorShape(out_shape)  # 4D

    def get_config(self):
        base_config = super(SpatialTransformLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class NeuralNetwork(object):

    def __init__(self, window=0.0625, step=0.03125, fs=160, channel_num=64):
        """
        Constructor for NN

        :param window: eeg window length in seconds
        :param step: window shift parameter in seconds
        :param fs: sampling frequency
        :param channel_num: number of eeg channels
        """
        assert fs, "Sampling frequency can't be {}".format(fs)
        self._window = window  # in seconds
        self._step = step  # in seconds
        self._fs = fs
        self._ch_num = channel_num
        self._model_built = False

    def set_window(self, window, step, fs=None):
        assert not self._model_built, "Can not change window parameters after building a model!"
        self._window = window
        self._step = step
        if fs:
            self._fs = fs

    def set_sampling_fequency(self, fs):
        assert not self._model_built, "Can not change sampling frequency after building a model!"
        self._fs = fs

    def set_channel_num(self, channel_num):
        self._ch_num = channel_num

    def _set_initial_dataset(self, filenames, shuffle_buffer_size=10000):
        """
        Creates dataset: multiple windows from one trigger window

        :param filenames: trigger windows
        :param shuffle_buffer_size: shuffle tiggers with given size
        :return: new tf.Dataset
        """
        dataset = tf.data.TFRecordDataset(filenames)

        # Shuffle the dataset
        if shuffle_buffer_size:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

        # example proto decode
        def _set_input_data(example_proto):
            keys_to_features = {DATA: tf.FixedLenFeature([], tf.string),
                                'm': tf.FixedLenFeature([], tf.int64),
                                'n': tf.FixedLenFeature([], tf.int64),
                                RECORD_TYPE_LABEL: tf.FixedLenFeature([], tf.int64),
                                TASK_LABEL: tf.FixedLenFeature([], tf.int64)}

            sample = tf.parse_single_example(example_proto, keys_to_features)
            data = tf.decode_raw(sample[DATA], tf.float32)
            shape = [sample['m'], sample['n']]
            data = tf.reshape(data, shape)

            return [data, [sample[RECORD_TYPE_LABEL], sample[TASK_LABEL]]]  # todo: one-hot encode?

        # Parse the record into tensors.
        dataset = dataset.map(_set_input_data)

        window = int(self._window * self._fs)  # in data points
        step = int(self._step * self._fs)  # in data points

        def _cut_to_data_epochs(data, label):  # todo: One label!
            channel, n = data.shape
            num = int((n - window) // step) + 1

            # one hot encode:
            assert len(label) == len(
                ONE_HOT_LIST), "Label number and one hot encoder length are not equal. Create new dataset!"
            output = list()
            for i, lab in enumerate(label):
                one_hot = [0] * len(ONE_HOT_LIST[i])
                one_hot[lab] = 1
                one_hot = [one_hot] * num
                output.append(np.array(one_hot, dtype=np.float32))

            # labels = [label] * num
            wind_data = [data[:, i * step:i * step + window] for i in range(num)]
            # return np.array(wind_data, dtype=np.float32), np.array(labels, dtype=np.float32)
            output.insert(0, np.array(wind_data, dtype=np.float32))
            return tuple(output)
            # return np.array(wind_data, dtype=np.float32), tuple(output)

        dataset = dataset.map(
            map_func=lambda input, label: tf.py_function(func=_cut_to_data_epochs, inp=[input, label],
                                                     Tout=[tf.float32, tf.float32, tf.float32]))
        # dataset = dataset.flat_map(lambda data, labels: tf.data.Dataset().zip((
        #     tf.data.Dataset().from_tensor_slices(data),
        #     tf.data.Dataset().from_tensor_slices(labels))
        # ))
        dataset = dataset.flat_map(lambda data, label0, label1: tf.data.Dataset().zip((
            tf.data.Dataset().from_tensor_slices(data),
            tf.data.Dataset().from_tensor_slices(label0),
            tf.data.Dataset().from_tensor_slices(label1))
        ))
        return dataset

    def _test_dataset(self, dataset):
        iterator = dataset.make_one_shot_iterator()
        next_data = iterator.get_next()
        ind = 0

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            try:
                # Keep extracting data till TFRecord is exhausted
                while True:
                    eeg, label1, label2 = sess.run(next_data)
                    print("{} data shape:{}, label1 {}, label2 {}".format(ind, eeg.shape, label1, label2))
                    ind += 1
            except tf.errors.OutOfRangeError:
                pass

    def _build_keras_model(self, conv_2d_filters=[32, 64, 128], lstm_layers=2):  # todo: continue...
        assert self._ch_num == 64, 'Spatial transformaiton is not defined for {} channels'.format(self._ch_num)
        self._model_built = True
        data_points = int(self._window * self._fs)

        main_input = keras.Input(shape=(self._ch_num, data_points), name='main_input')
        x = keras.layers.BatchNormalization()(main_input)  # TODO: Normalisation here or in preprocess?
        x = SpatialTransformLayer((10, 11, data_points))(x)

        # split data
        list_x = keras.layers.Lambda(
            lambda t: [t[:, :, :, i] for i in range(keras.backend.int_shape(t)[-1])])(x)

        # 2D convolution block
        cnn_out = list()
        for x in list_x:
            x = keras.layers.Lambda(lambda t: tf.keras.backend.expand_dims(t))(x)
            for filter_num in conv_2d_filters:
                x = keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), padding='same',
                                        activation=tf.nn.leaky_relu)(x)
            x = keras.layers.Flatten()(x)
            x = keras.layers.Dense(units=1024, activation='sigmoid')(x)  # activation function is not defined in article
            x = keras.layers.Lambda(lambda t: tf.keras.backend.expand_dims(t))(x)  # TODO: remove?
            cnn_out.append(x)

        # LSTM
        x = keras.layers.Concatenate()(cnn_out)
        # x = cnn_out
        for i in range(lstm_layers):
            ret_seq = i < lstm_layers - 1
            if tf.test.is_built_with_cuda():
                x = keras.layers.CuDNNLSTM(data_points, return_sequences=ret_seq)(x)
            else:
                x = keras.layers.LSTM(data_points, return_sequences=ret_seq)(x)

        x = keras.layers.Dense(1024, activation='sigmoid')(x)

        outputs = list()
        for one_hot in ONE_HOT_LIST:
            outputs.append(keras.layers.Dense(len(one_hot), activation='softmax')(x))

        model = keras.Model(inputs=[main_input], outputs=outputs)
        return model

    def offline_training(self, filenames):  # TODO: continue
        assert len(filenames) == 3, "3 set of filenames are required. Got {} instead".format(len(filenames))
        train_dataset = self._set_initial_dataset(filenames.get(TRAIN), shuffle_buffer_size=None)
        val_dataset = self._set_initial_dataset(filenames.get(VALIDATION))
        test_dataset = self._set_initial_dataset(filenames.get(TEST))

        # self._test_dataset(train_dataset)
        # return
        model = self._build_keras_model()
        model.compile(optimizer=tf.train.AdamOptimizer(0.0001),
                      loss=['categorical_crossentropy'] * len(ONE_HOT_LIST),
                      metrics=['accuracy'] * len(ONE_HOT_LIST))
        # tbCallBack = keras.callbacks.TensorBoard(log_dir='tflog', histogram_freq=0,
        #                                          write_graph=True, write_images=True)
        # tbCallBack.set_model(model)
        print("Done")
        batch = 32
        train_dataset = train_dataset.batch(batch)
        train_dataset = train_dataset.repeat()
        # print(int(len(filenames.get(TRAIN)) // batch))
        model.fit(train_dataset, epochs=2, steps_per_epoch=int(len(filenames.get(TRAIN)) // batch))
        # todo: changed modul in: tensorflow/python/keras/engine/training_utils.py, line: 215


if __name__ == '__main__':
    base_dir = "D:/BCI_stuff/databases/"  # MTA TTK
    # base_dir = 'D:/Csabi/'  # Home
    # base_dir = "D:/Users/Csabi/data"  # ITK
    # base_dir = "/home/csabi/databases/"  # linux

    from BCISystem import BCISystem

    bci = BCISystem(base_dir)
    bci.offline_processing()

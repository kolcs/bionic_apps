from ..utils import load_pickle_data, _create_binary_label
from numpy.random import shuffle


class DataHandler:  # todo: move to TFRecord - https://www.tensorflow.org/guide/data#consuming_tfrecord_data

    def __init__(self, file_list, label_encoder, binary_classification=False,
                 shuffle_epochs=True, epoch_buffer=5, shuffle_windows=True):
        """Data handler for data, which is greater than the memory capacity.

        This data handler integrates the data with the tensorflow dataset API.

        Parameters
        ----------
        file_list : list of str
            list of epoch filenames
        label_encoder : LabelEncoder
            In case of binary classification converts the labels.
        binary_classification : bool
            To use or not the label converter.
        shuffle_epochs : bool
            If yes, the order of epochs will be shuffled.
        epoch_buffer : int
            How much epochs should be preloaded. Used for window shuffling.
        shuffle_windows : bool
            If yes, the order of windows will be shuffled in the epoch buffer.
        """
        if shuffle_epochs:
            shuffle(file_list)
        self._file_list = list(file_list)
        self._label_encoder = label_encoder
        self._binary_classification = binary_classification
        window_list = load_pickle_data(self._file_list[0])
        data, label = window_list[0]
        self._win_num = len(window_list)
        self._data_shape = data.shape
        self._label_shape = label.shape
        self._preload_epoch_num = epoch_buffer
        self._shuffle_windows = shuffle_windows
        self._data_list = list()

    def _load_epoch(self, filename):
        window_list = load_pickle_data(filename)
        for data, labels in window_list:
            if self._binary_classification:
                labels = [_create_binary_label(label) for label in labels]
            labels = self._label_encoder.transform(labels)
            self._data_list.append((data, labels))

    def _generate_data(self):
        for i in range(self._preload_epoch_num):
            self._load_epoch(self._file_list[i])
        for i in range(self._preload_epoch_num, len(self._file_list)):
            self._load_epoch(self._file_list[i])
            if self._shuffle_windows:
                shuffle(self._data_list)
            for _ in range(self._win_num):
                yield self._data_list.pop(0)

        while len(self._data_list) > 0:  # all files loaded
            yield self._data_list.pop(0)

    def get_tf_dataset(self):
        """Getting tensorflow Dataset from generator."""
        # dataset = tf.data.Dataset.from_tensor_slices(self._file_list)  # or generator
        # dataset = dataset.map(self._load_data)
        import tensorflow as tf
        dataset = tf.data.Dataset.from_generator(
            self._generate_data,
            output_types=(tf.float32, tf.int32),
            output_shapes=(self._data_shape, self._label_shape)
        )
        return dataset

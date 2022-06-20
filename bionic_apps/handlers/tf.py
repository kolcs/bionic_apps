import tensorflow as tf

from .hdf5 import HDF5Dataset


# @kolcs: move to TFRecord? - https://www.tensorflow.org/guide/data#consuming_tfrecord_data
def get_tf_dataset(db, y, indexes):
    """Generating tensorflow dataset from hdf5 database

    Returns a dataset from the tensorflow dataset API.

    Parameters
    ----------
    db : HDF5Dataset
        HDF5 database from where data can be loaded.
    y : ndarray
        Labels converted to integer numbers corresponding to the whole dataset.
    indexes : ndarray or list
        Iterable index order, which specifies the loadin order of the data

    Returns
    -------
    tf.data.Dataset
        Tensorflow Dataset
    """
    assert len(y) > max(indexes), \
        'Some indexes are higher than the length of the y variable. ' \
        'This error may occur when the indexes are correspond to the whole database ' \
        'and the y are filtered to a subject.'

    def generate_data():
        for i in indexes:
            x = db.get_data(i)
            yield x, y[i]

    data_shape = db.get_data(indexes[0]).shape
    label_shape = y[indexes[0]].shape

    dataset = tf.data.Dataset.from_generator(
        generate_data,
        output_types=(tf.float32, tf.int32),
        output_shapes=(data_shape, label_shape)
    )
    return dataset

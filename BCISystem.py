from config import *
from preprocess import OfflineDataPreprocessor
from ai.neural_network import NeuralNetwork


# TODO: build NN

class BCISystem(object):

    def __init__(self, base_dir="", window=0.0625, step=0.03125, fs=160): # TODO: check base dir
        """
        Constructor for BCI system

        :param base_dir: base directory of databases
            in case of offline process - source folder of database
            in case of online process - where to save database
        :param window: eeg window length in seconds
        :param step: window shift parameter in seconds
        :param fs: sampling fequency
        """
        self._base_dir = base_dir
        self._window = window  # in seconds
        self._step = step  # in seconds
        self._fs = fs

    def offline_processing(self):
        """
        This is the main function which starts the offline processing
        """
        directory = self._base_dir + DIR_TF_RECORDS

        filenames = dict()
        filenames[TRAIN] = OfflineDataPreprocessor.get_filenames_in(directory + DIR_TRAIN, ext=F_EXT_TF_RECORD)
        filenames[VALIDATION] = OfflineDataPreprocessor.get_filenames_in(directory + DIR_VALIDATION, ext=F_EXT_TF_RECORD)
        filenames[TEST] = OfflineDataPreprocessor.get_filenames_in(directory + DIR_TEST, ext=F_EXT_TF_RECORD)

        nn = NeuralNetwork()
        nn.offline_training(filenames)

    def online_processing(self):
        raise NotImplementedError("Online processing is not implemented...")


if __name__ == '__main__':
    # base_dir = "D:/BCI_stuff/databases/"  # MTA TTK
    # base_dir = 'D:/Csabi/'  # Home
    base_dir = "D:/Users/Csabi/data/"  # ITK
    # base_dir = "/home/csabi/databases/"  # linux

    bci = BCISystem(base_dir)
    bci.offline_processing()

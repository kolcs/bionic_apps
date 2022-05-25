import unittest
from pathlib import Path
from time import time

from mne import EpochsArray, Epochs
from numpy import ndarray

from config import TTK_DB
from preprocess import init_base_config, get_epochs_from_files, DataLoader, ArtefactFilter


class TestFaster(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._subject = 24
        cls.faster = ArtefactFilter(apply_frequency_filter=False)

    def _check_method(self, epochs):
        n_epochs = len(epochs)
        split_ind = n_epochs - 1
        offline_epochs = epochs[:split_ind].copy()
        online_epochs = epochs[split_ind:].copy()

        filt_epochs = self.faster.offline_filter(offline_epochs)
        self.assertIn(type(filt_epochs), [Epochs, EpochsArray])
        print('\n*******************************************************'
              '\nOnline FASTER'
              '\n*******************************************************')
        tic = time()
        filt_epochs = self.faster.online_filter(online_epochs.get_data())
        toc = time()
        self.assertIsInstance(filt_epochs, ndarray)
        self.assertLess(toc - tic, .1, 'Online FASTER is not fast enough...')
        print(f'Total time spent in lsl: {toc - tic}')

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(TTK_DB.DIR).exists(),
                         'Data for TTK does not exists. Can not test it.')
    def test_ttk(self):
        loader = DataLoader('..').use_ttk_db()
        file = loader.get_filenames_for_subject(self._subject)
        epochs = get_epochs_from_files(file, TTK_DB.TRIGGER_TASK_CONVERTER, epoch_tmin=0, epoch_tmax=4,
                                       event_id=TTK_DB.TRIGGER_EVENT_ID, preload=True, prefilter_signal=True,
                                       order=5, l_freq=1, h_freq=45)
        self._check_method(epochs)


if __name__ == '__main__':
    unittest.main()

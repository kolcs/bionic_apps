import unittest
from multiprocessing import Process
from pathlib import Path

from BCISystem import BCISystem, Databases, FeatureType, XvalidateMethod
from config import Physionet, Game_ParadigmD, DIR_FEATURE_DB
from online.DataSender import run as send_online_data
from preprocess import init_base_config, recursive_delete_folder


# @unittest.skip("Not interested")
class TestOfflineBciSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        path = Path(init_base_config('..'))
        path = path.joinpath(DIR_FEATURE_DB)
        if path.exists():
            print('Removing old files. It may take longer...')
            recursive_delete_folder(str(path))
        cls.subj = 1

    def setUp(self):
        self.bci = BCISystem()

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Physionet.DIR).exists(),
                         'Data for Physionet does not exists. Can not test it.')
    def test_physionet_subject_x_val(self):
        feature_extraction = dict(
            feature_type=FeatureType.FFT_POWER,
            fft_low=14, fft_high=30
        )
        self.assertIsNone(self.bci.offline_processing(
            Databases.PHYSIONET, feature_params=feature_extraction,
            epoch_tmin=0, epoch_tmax=4,
            window_length=1, window_step=.1,
            method=XvalidateMethod.SUBJECT,
            subject_list=self.subj,
        ))

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Game_ParadigmD.DIR).exists(),
                         'Data for Game_paradigmD does not exists. Can not test it.')
    def test_subject_x_val_fft_power(self):
        feature_extraction = dict(
            feature_type=FeatureType.FFT_POWER,
            fft_low=14, fft_high=30
        )
        self.assertIsNone(self.bci.offline_processing(
            Databases.GAME_PAR_D, feature_params=feature_extraction,
            epoch_tmin=0, epoch_tmax=4,
            window_length=1, window_step=.1,
            method=XvalidateMethod.SUBJECT,
            subject_list=self.subj,
        ))

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Game_ParadigmD.DIR).exists(),
                         'Data for Game_paradigmD does not exists. Can not test it.')
    def test_subject_x_val_fft_range(self):
        feature_extraction = dict(
            feature_type=FeatureType.FFT_RANGE,
            fft_low=14, fft_high=30, fft_step=2, fft_width=2
        )
        self.assertIsNone(self.bci.offline_processing(
            Databases.GAME_PAR_D, feature_params=feature_extraction,
            epoch_tmin=0, epoch_tmax=4,
            window_length=1, window_step=.1,
            method=XvalidateMethod.SUBJECT,
            subject_list=self.subj,
        ))

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Game_ParadigmD.DIR).exists(),
                         'Data for Game_paradigmD does not exists. Can not test it.')
    def test_subject_x_val_multi_fft_power(self):
        feature_extraction = dict(
            feature_type=FeatureType.MULTI_FFT_POWER,
            fft_ranges=[(14, 36), (18, 32), (18, 36), (22, 28),
                        (22, 32), (22, 36), (26, 32), (26, 36)]
        )
        self.assertIsNone(self.bci.offline_processing(
            Databases.GAME_PAR_D, feature_params=feature_extraction,
            epoch_tmin=0, epoch_tmax=4,
            window_length=1, window_step=.1,
            method=XvalidateMethod.SUBJECT,
            subject_list=self.subj,
        ))


@unittest.skipUnless(Path(init_base_config('..')).joinpath('Game', 'paradigmD', 'subject2', 'game01.vhdr').exists(),
                     "No files found for live eeg simulation.")
class TestOnlineBci(unittest.TestCase):
    _live_eeg_emulator = Process()

    @classmethod
    def setUpClass(cls):
        cls._path = Path(init_base_config())
        cls._path = cls._path.joinpath('Game', 'paradigmD', 'subject2')
        cls._live_eeg_emulator = Process(target=send_online_data,
                                         kwargs=dict(filename=cls._path.joinpath('game01.vhdr'), get_labels=False,
                                                     add_extra_data=True))
        cls._live_eeg_emulator.start()

    @classmethod
    def tearDownClass(cls):
        cls._live_eeg_emulator.terminate()

    def setUp(self):
        self.bci = BCISystem()

    def test_play_game(self):
        test_time_out = 2  # sec
        feature_extraction = dict(
            feature_type=FeatureType.MULTI_FFT_POWER,
            fft_ranges=[(14, 36), (18, 32), (18, 36), (22, 28),
                        (22, 32), (22, 36), (26, 32), (26, 36)]
        )
        self.assertIsNone(self.bci.play_game(
            Databases.GAME_PAR_D, feature_params=feature_extraction,
            epoch_tmin=0, epoch_tmax=4,
            window_length=1, pretrain_window_step=.1,
            train_file=str(self._path.joinpath('rec01.vhdr')),
            time_out=test_time_out
        ))


if __name__ == '__main__':
    unittest.main()

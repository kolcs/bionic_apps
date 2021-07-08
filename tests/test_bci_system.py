import unittest
from multiprocessing import Process
from pathlib import Path

from BCISystem import BCISystem, Databases, FeatureType, XvalidateMethod, SubjectHandle
from online.DataSender import run as send_online_data
from preprocess import init_base_config, DataLoader
from tests.utils import cleanup_fastload_data, AVAILABLE_DBS


# @unittest.skip("Not interested")
class TestOfflineBciSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cleanup_fastload_data()

    def setUp(self):
        self.bci = BCISystem()

    def _run_test(self, feature_extraction, db_config_ver=-1, subject_handle=SubjectHandle.INDEPENDENT_DAYS,
                  mimic_online=False, filter_params=None, xval_method=XvalidateMethod.SUBJECT,
                  subj_n_fold_num=5, subj_num=1):
        for db_name in AVAILABLE_DBS:
            with self.subTest(f'Database: {db_name.name}'):
                loader = DataLoader('..', subject_handle=subject_handle).use_db(db_name, db_config_ver)
                subject = loader.get_subject_list()[:subj_num]
                self.assertIsNone(self.bci.offline_processing(
                    db_name, db_config_ver=db_config_ver,
                    feature_params=feature_extraction,
                    epoch_tmin=0, epoch_tmax=4,
                    window_length=2, window_step=.1,
                    method=xval_method,
                    subject_list=subject,
                    subject_handle=subject_handle,
                    filter_params=filter_params,
                    mimic_online_method=mimic_online,
                    subj_n_fold_num=subj_n_fold_num,
                ))

    def test_subject_x_val_mimic_online(self):
        filter_params = dict(  # required for FASTER artefact filter
            order=5, l_freq=1, h_freq=45
        )
        feature_extraction = dict(
            feature_type=FeatureType.AVG_FFT_POWER,
            fft_low=14, fft_high=30
        )
        self._run_test(feature_extraction, mimic_online=True, filter_params=filter_params)

    def test_subject_x_val_fft_power(self):
        feature_extraction = dict(
            feature_type=FeatureType.AVG_FFT_POWER,
            fft_low=14, fft_high=30
        )
        self._run_test(feature_extraction, subj_num=2)

    def test_subject_x_val_fft_power_old_db_config(self):
        feature_extraction = dict(
            feature_type=FeatureType.AVG_FFT_POWER,
            fft_low=14, fft_high=30
        )
        self._run_test(feature_extraction, db_config_ver=0)

    def test_subject_x_val_fft_range(self):
        feature_extraction = dict(
            feature_type=FeatureType.FFT_RANGE,
            fft_low=14, fft_high=30
        )
        self._run_test(feature_extraction)

    def test_subject_x_val_multi_fft_power(self):
        feature_extraction = dict(
            feature_type=FeatureType.MULTI_AVG_FFT_POW,
            fft_ranges=[(14, 36), (18, 32), (18, 36), (22, 28),
                        (22, 32), (22, 36), (26, 32), (26, 36)]
        )
        self._run_test(feature_extraction)

    def test_cross_subject_fft_power(self):
        feature_extraction = dict(
            feature_type=FeatureType.AVG_FFT_POWER,
            fft_low=14, fft_high=30
        )
        self._run_test(feature_extraction, xval_method=XvalidateMethod.CROSS_SUBJECT,
                       subj_n_fold_num=2, subj_num=2)


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
            feature_type=FeatureType.MULTI_AVG_FFT_POW,
            fft_ranges=[(14, 36), (18, 32), (18, 36), (22, 28),
                        (22, 32), (22, 36), (26, 32), (26, 36)]
        )
        self.assertIsNone(self.bci.play_game(
            Databases.GAME_PAR_D, feature_params=feature_extraction,
            epoch_tmin=0, epoch_tmax=4,
            window_length=2, pretrain_window_step=.1,
            train_file=str(self._path.joinpath('rec01.vhdr')),
            time_out=test_time_out
        ))


if __name__ == '__main__':
    unittest.main()

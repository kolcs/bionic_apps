import unittest
from multiprocessing import Process
from pathlib import Path

from sklearn.ensemble import ExtraTreesClassifier

from bionic_apps.ai import ClassifierType
from bionic_apps.external_connections.lsl.DataSender import run as send_online_data
from bionic_apps.feature_extraction import FeatureType, get_hugines_transfromer
from bionic_apps.feature_extraction import eeg_bands
from bionic_apps.games.braindriver.main_game_start import start_brain_driver_control_system
from bionic_apps.offline_analyses import test_db_within_subject, test_db_cross_subject
from bionic_apps.tests.utils import cleanup_fastload_data, AVAILABLE_DBS
from bionic_apps.utils import init_base_config


# @unittest.skip("Not interested")
class TestOfflineBciSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cleanup_fastload_data()

    def _run_eegdb_within_subj_test(self, **kwargs):
        for db_name in AVAILABLE_DBS:
            with self.subTest(f'Database: {db_name.name}'):
                test_db_within_subject(db_name,
                                       window_len=2, window_step=1,
                                       do_artefact_rejection=True,
                                       fast_load=True, log_file='out.csv',
                                       subjects=2,
                                       **kwargs)

    def test_eegdb_within_subj_multi_svm(self):
        band = eeg_bands['range40'].copy()
        kwargs = dict(
            feature_kwargs=band,
            feature_type=band.pop('feature_type'),
            filter_params=dict(  # required for FASTER artefact filter
                order=5, l_freq=1, h_freq=45
            ),
            classifier_type=ClassifierType.VOTING_SVM,
        )
        self._run_eegdb_within_subj_test(**kwargs)

    def test_eegdb_within_subj_custom(self):
        kwargs = dict(
            feature_type=FeatureType.USER_PIPELINE,
            classifier_type=ClassifierType.USER_DEFINED,
            feature_kwargs=dict(pipeline=get_hugines_transfromer()),
            classifier_kwargs=dict(
                classifier=ExtraTreesClassifier(n_estimators=250, n_jobs=-2)
            ),
            filter_params=dict(  # required for FASTER artefact filter
                order=5, l_freq=1, h_freq=45
            ),

        )
        self._run_eegdb_within_subj_test(**kwargs)

    def _run_eegdb_cross_subj_test(self, **kwargs):
        for db_name in AVAILABLE_DBS:
            with self.subTest(f'Database: {db_name.name}'):
                test_db_cross_subject(db_name,
                                      window_len=2, window_step=1,
                                      do_artefact_rejection=True,
                                      fast_load=True, log_file='out.csv',
                                      subjects=5, leave_out_n_subjects=2,
                                      **kwargs)

    def test_eegdb_cross_subj_eegnet_no_val(self):
        kwargs = dict(
            feature_type=FeatureType.RAW,
            classifier_type=ClassifierType.EEG_NET,
            classifier_kwargs=dict(
                epochs=2
            ),
            filter_params=dict(  # required for FASTER artefact filter
                order=5, l_freq=1, h_freq=45
            ),

        )
        self._run_eegdb_cross_subj_test(**kwargs)

    def test_eegdb_cross_subj_eegnet_val(self):
        kwargs = dict(
            feature_type=FeatureType.RAW,
            classifier_type=ClassifierType.EEG_NET,
            classifier_kwargs=dict(
                epochs=2, validation_split=.1
            ),
            filter_params=dict(  # required for FASTER artefact filter
                order=5, l_freq=1, h_freq=45
            ),
        )
        self._run_eegdb_cross_subj_test(**kwargs)


@unittest.skipUnless(
    Path(init_base_config()).joinpath('Game', 'paradigmD', 'subject2', 'game01.vhdr').exists(),
    "No files found for live eeg simulation.")
class TestBrainDriverBci(unittest.TestCase):
    _live_eeg_emulator = Process()
    _path = Path(init_base_config())
    test_time_out = 5  # sec

    @classmethod
    def setUpClass(cls):
        cleanup_fastload_data()
        cls._path = cls._path.joinpath('Game', 'paradigmD', 'subject2')
        cls._live_eeg_emulator = Process(target=send_online_data,
                                         kwargs=dict(filenames=str(cls._path.joinpath('game01.vhdr')), get_labels=False,
                                                     add_extra_data=True))
        cls._live_eeg_emulator.start()

    @classmethod
    def tearDownClass(cls):
        cls._live_eeg_emulator.terminate()

    def _test_play_game(self, feature_type, classifier_type, use_best_clf=False,
                        feature_kwargs=None, classifier_kwargs=None):
        start_brain_driver_control_system(feature_type, classifier_type,
                                          eeg_files=str(self._path.joinpath('rec01.vhdr')),
                                          time_out=self.test_time_out,
                                          use_game_logger=False, make_opponents=False,
                                          use_best_clf=use_best_clf,
                                          feature_kwargs=feature_kwargs,
                                          classifier_kwargs=classifier_kwargs,
                                          do_artefact_rejection=False)

    def test_play_game_multi_svm_best_clf(self):
        feature_type = FeatureType.FFT_RANGE
        classifier_type = ClassifierType.VOTING_SVM
        feature_kwargs = dict(
            fft_low=2, fft_high=40
        )
        self._test_play_game(feature_type, classifier_type, feature_kwargs=feature_kwargs)

    def test_play_game_multi_svm(self):
        feature_type = FeatureType.FFT_RANGE
        classifier_type = ClassifierType.VOTING_SVM
        feature_kwargs = dict(
            fft_low=2, fft_high=40
        )
        self._test_play_game(feature_type, classifier_type, feature_kwargs=feature_kwargs,
                             use_best_clf=False)

    def test_play_game_eeg_net_best_clf(self):
        feature_type = FeatureType.RAW
        classifier_type = ClassifierType.EEG_NET
        classifier_kwargs = dict(epochs=1)
        self._test_play_game(feature_type, classifier_type,
                             classifier_kwargs=classifier_kwargs)

    def test_play_game_eeg_net(self):
        feature_type = FeatureType.RAW
        classifier_type = ClassifierType.EEG_NET
        classifier_kwargs = dict(epochs=1)
        self._test_play_game(feature_type, classifier_type, use_best_clf=False,
                             classifier_kwargs=classifier_kwargs)


if __name__ == '__main__':
    unittest.main()

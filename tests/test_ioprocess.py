import unittest
from pathlib import Path
from shutil import rmtree

from config import Physionet, Game_ParadigmD, DIR_FEATURE_DB
from preprocess import init_base_config, OfflineDataPreprocessor, FeatureType


class TestPreprocessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._path = Path(init_base_config('..'))
        cls._subject = 1
        path = str(cls._path.joinpath(DIR_FEATURE_DB))
        rmtree(path)

    def setUp(self):
        self.epoch_proc = OfflineDataPreprocessor(self._path, subject=self._subject)

    def _check_method(self):
        feature_extraction = dict(
            feature_type=FeatureType.FFT_POWER,
            fft_low=14, fft_high=30
        )
        self.epoch_proc.run(**feature_extraction)
        self.assertIsInstance(self.epoch_proc.get_processed_db_source(), dict)
        self.assertGreater(len(self.epoch_proc.get_processed_db_source()), 0)

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Physionet.DIR).exists(),
                         'Data for Physionet does not exists. Can not test it.')
    def test_1_physionet(self):
        self.epoch_proc.use_physionet()
        self._check_method()

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Game_ParadigmD.DIR).exists(),
                         'Data for Game_paradigmD does not exists. Can not test it.')
    def test_2_game_paradigmD(self):
        self.epoch_proc.use_game_par_d()
        self._check_method()

    def test_3_fast_load(self):
        self.epoch_proc.use_physionet()
        self.test_2_game_paradigmD()

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Game_ParadigmD.DIR).exists(),
                         'Data for Game_paradigmD does not exists. Can not test it.')
    def test_4_data_update(self):
        self.epoch_proc = OfflineDataPreprocessor(self._path, subject=self._subject + 1)
        self.test_2_game_paradigmD()
        self.assertGreater(len(self.epoch_proc.get_processed_db_source()), 1)


# @unittest.skipUnless(Path(init_base_config('..')).joinpath(Physionet.DIR).exists(),
#                      'Data for Physionet does not exists. Can not test it.')
# class TestSubjectKFold(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         path = Path(init_base_config('..'))
#         cls.kfold_num = 5
#         cls.epoch_proc = EpochPreprocessor(path, subject=list(range(1, cls.kfold_num + 1)),
#                                            fast_load=False).use_physionet()
#         cls.epoch_proc.run()
#         cls.subj_k_fold = SubjectKFold(cls.epoch_proc, cls.kfold_num, equalize_data=True)
#
#     def _check_method(self, ans):
#         self.assertEqual(len(ans), self.kfold_num)
#         test = ans[0][1]
#         keys = list(test)
#         try:
#             self.assertIsInstance(test[keys[0]][0], mne_epochs.Epochs)
#         except AssertionError:
#             self.assertIsInstance(test[keys[0]][0], mne_epochs.EpochsFIF)
#
#     def test_subject_kfold_split_subject_physio(self):
#         ans = list()
#         for train, test, subj in self.subj_k_fold.split_subject_data(1):
#             ans.append((train, test))
#         self._check_method(ans)
#
#     def test_subject_kfold_split_physio(self):
#         ans = list()
#         for train, test, test_subj in self.subj_k_fold.split_db():
#             ans.append((train, test))
#         self._check_method(ans)


if __name__ == '__main__':
    unittest.main()

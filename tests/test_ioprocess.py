import unittest
from pathlib import Path
from shutil import rmtree

from mne import epochs as mne_epochs

from config import Physionet, Game_ParadigmD
from preprocess import init_base_config, OfflineDataPreprocessor, SubjectKFold


class TestPreprocessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._path = Path(init_base_config('..'))
        cls._subject = 1

    def setUp(self):
        self.epoch_proc = OfflineDataPreprocessor(self._path, subject=self._subject, fast_load=False)

    def _check_method(self):
        self.epoch_proc.run()
        self.assertGreater(len(self.epoch_proc.get_subjects()), 0)
        ans = self.epoch_proc.get_data_for_subject_split(self._subject)
        self.assertIsInstance(ans, dict)
        keys = list(ans)
        try:
            self.assertIsInstance(ans[keys[0]][0], mne_epochs.Epochs)
        except AssertionError:
            self.assertIsInstance(ans[keys[0]][0], mne_epochs.EpochsFIF)

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Physionet.DIR).exists(),
                         'Data for Physionet does not exists. Can not test it.')
    def test_physionet(self):
        self.epoch_proc.use_physionet()
        self._check_method()

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Game_ParadigmD.DIR).exists(),
                         'Data for Game_paradigmD does not exists. Can not test it.')
    def test_game_paradigmD(self):
        self.epoch_proc.use_game_par_d()
        self._check_method()

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Game_ParadigmD.DIR).exists(),
                         'Data for Game_paradigmD does not exists. Can not test it.')
    def test_fast_load_without_data(self):
        path = str(Path(self._path).joinpath('tmp'))
        rmtree(path)
        self.epoch_proc = EpochPreprocessor(self._path, subject=self._subject, fast_load=True)
        self.epoch_proc.use_game_par_d()
        self._check_method()

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Game_ParadigmD.DIR).exists(),
                         'Data for Game_paradigmD does not exists. Can not test it.')
    def test_fast_load_with_data(self):
        self.epoch_proc.use_game_par_d()
        self.epoch_proc.run()
        self.epoch_proc = EpochPreprocessor(self._path, subject=self._subject, fast_load=True)
        self.epoch_proc.use_game_par_d()
        self._check_method()


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

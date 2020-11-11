import unittest
from pathlib import Path
from shutil import rmtree

from numpy import ndarray as ndarray

from config import Physionet, Game_ParadigmD, DIR_FEATURE_DB
from preprocess import init_base_config, OfflineDataPreprocessor, FeatureType, SubjectKFold, load_pickle_data


class TestPreprocessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._path = Path(init_base_config('..'))
        cls._subject = 1
        path = str(cls._path.joinpath(DIR_FEATURE_DB))
        rmtree(path)
        cls.epoch_proc = OfflineDataPreprocessor(cls._path, subject=cls._subject)

    # def setUp(self):
    #     pass

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
        self.test_1_physionet()
        self.test_2_game_paradigmD()

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Game_ParadigmD.DIR).exists(),
                         'Data for Game_paradigmD does not exists. Can not test it.')
    def test_4_data_update(self):
        self.epoch_proc = OfflineDataPreprocessor(self._path, subject=self._subject + 1)
        self.test_2_game_paradigmD()
        self.assertGreater(len(self.epoch_proc.get_processed_db_source()), 1)


class TestSubjectKFold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._path = Path(init_base_config('..'))
        cls.kfold_num = 5
        cls.epoch_proc = OfflineDataPreprocessor(cls._path, subject=list(range(1, cls.kfold_num + 1)))

    # def setUp(self):
    #     pass

    def _run_epoch_proc(self):
        feature_extraction = dict(
            feature_type=FeatureType.FFT_POWER,
            fft_low=14, fft_high=30
        )
        self.epoch_proc.run(**feature_extraction)

    def _check_method(self, ans):
        self.assertEqual(len(ans), self.kfold_num)

        train, test = ans[0]
        self.assertTrue(Path(train[0]).exists())
        self.assertTrue(Path(test[0]).exists())

        data = load_pickle_data(train[0])
        self.assertIsInstance(data[0], ndarray)
        self.assertIn(type(data[1]), [str, int, float, list, ndarray])

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Physionet.DIR).exists(),
                         'Data for Physionet does not exists. Can not test it.')
    def test_subject_kfold_split_subject_physio(self):
        subj_kfold = SubjectKFold(self.epoch_proc.use_physionet(), self.kfold_num)
        self._run_epoch_proc()
        ans = list()
        for train, test, val, subj in subj_kfold.split(1):
            ans.append((train, test))
        self._check_method(ans)

    def test_subject_kfold_split_physio(self):
        subj_kfold = SubjectKFold(self.epoch_proc.use_physionet(), self.kfold_num)
        self._run_epoch_proc()
        ans = list()
        for train, test, val, subj in subj_kfold.split():
            ans.append((train, test))
        self._check_method(ans)


if __name__ == '__main__':
    unittest.main()

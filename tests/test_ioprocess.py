import unittest
from pathlib import Path

from numpy import ndarray
from sklearn.preprocessing import LabelEncoder

from config import Physionet, Game_ParadigmC, Game_ParadigmD, DIR_FEATURE_DB
from preprocess import init_base_config, OfflineDataPreprocessor, FeatureType, SubjectKFold, load_pickle_data, \
    DataHandler, recursive_delete_folder


class TestPreprocessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._path = Path(init_base_config('..'))
        cls._subject = 1
        path = cls._path.joinpath(DIR_FEATURE_DB)
        if path.exists():
            print('Removing old files. It may take longer...')
            recursive_delete_folder(str(path))
        cls.epoch_proc = OfflineDataPreprocessor(cls._path)

    # def setUp(self):
    #     pass

    def _check_method(self, subj):
        feature_extraction = dict(
            feature_type=FeatureType.FFT_POWER,
            fft_low=14, fft_high=30
        )
        self.epoch_proc.run(subject=subj, **feature_extraction)
        self.assertIsInstance(self.epoch_proc.get_processed_db_source(), dict)
        self.assertGreater(len(self.epoch_proc.get_processed_db_source()), 0)

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Physionet.DIR).exists(),
                         'Data for Physionet does not exists. Can not test it.')
    def test_1_physionet(self):
        self.epoch_proc.use_physionet()
        self._check_method(self._subject)

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Game_ParadigmD.DIR).exists(),
                         'Data for Game_paradigmD does not exists. Can not test it.')
    def test_2_game_paradigmD(self):
        self.epoch_proc.use_game_par_d()
        self._check_method(self._subject)

    def test_3_fast_load(self):
        self.test_1_physionet()
        self.test_2_game_paradigmD()

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Game_ParadigmD.DIR).exists(),
                         'Data for Game_paradigmD does not exists. Can not test it.')
    def test_4_data_update(self):
        # self.epoch_proc = OfflineDataPreprocessor(self._path, subject=self._subject + 1)
        self.epoch_proc.use_game_par_d()
        self._check_method(self._subject + 1)
        self.assertGreater(len(self.epoch_proc.get_processed_db_source()), 1)


class TestSubjectKFold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._path = Path(init_base_config('..'))
        cls.kfold_num = 5
        cls.epoch_proc = OfflineDataPreprocessor(cls._path)

    # def setUp(self):
    #     pass

    def _run_epoch_proc(self):
        feature_extraction = dict(
            feature_type=FeatureType.FFT_POWER,
            fft_low=14, fft_high=30
        )
        self.epoch_proc.run(subject=list(range(1, self.kfold_num + 1)), **feature_extraction)

    def _check_method(self, ans, kfold_num=None):
        if kfold_num is None:
            kfold_num = self.kfold_num
        self.assertEqual(len(ans), kfold_num)

        train, test = ans[0]
        self.assertTrue(Path(train[0]).exists())
        self.assertTrue(Path(test[0]).exists())

        window_list = load_pickle_data(train[0])
        data = window_list[0]
        self.assertIsInstance(data[0], ndarray)
        self.assertIn(type(data[1]), [str, int, float, list, ndarray])

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Physionet.DIR).exists(),
                         'Data for Physionet does not exists. Can not test it.')
    def test_split_subject_physio(self):
        subj_kfold = SubjectKFold(self.epoch_proc.use_physionet(), self.kfold_num)
        self._run_epoch_proc()
        ans = list()
        for train, test, val, subj in subj_kfold.split(1):
            ans.append((train, test))
        self._check_method(ans)

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Physionet.DIR).exists(),
                         'Data for Physionet does not exists. Can not test it.')
    def test_split_subject_physio_binarize(self):
        subj_kfold = SubjectKFold(self.epoch_proc.use_physionet(), self.kfold_num, binarize_db=True)
        self._run_epoch_proc()
        ans = list()
        for train, test, val, subj in subj_kfold.split(1):
            ans.append((train, test))
        self._check_method(ans)

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Physionet.DIR).exists(),
                         'Data for Physionet does not exists. Can not test it.')
    def test_cross_split_physio(self):
        subj_kfold = SubjectKFold(self.epoch_proc.use_physionet(), self.kfold_num)
        self._run_epoch_proc()
        ans = list()
        for train, test, val, subj in subj_kfold.split(cross_subject=True):
            ans.append((train, test))
        self._check_method(ans)

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Physionet.DIR).exists(),
                         'Data for Physionet does not exists. Can not test it.')
    def test_cross_split_one_physio(self):
        subj_kfold = SubjectKFold(self.epoch_proc.use_physionet(), self.kfold_num)
        self._run_epoch_proc()
        ans = list()
        for train, test, val, subj in subj_kfold.split(1, cross_subject=True):
            ans.append((train, test))
        self._check_method(ans, 1)

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Physionet.DIR).exists(),
                         'Data for Physionet does not exists. Can not test it.')
    def test_cross_split_physio_binarized(self):
        subj_kfold = SubjectKFold(self.epoch_proc.use_physionet(), self.kfold_num, binarize_db=True)
        self._run_epoch_proc()
        ans = list()
        for train, test, val, subj in subj_kfold.split(cross_subject=True):
            ans.append((train, test))
        self._check_method(ans)

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Game_ParadigmC.DIR).exists(),
                         'Data for Game_ParadigmC does not exists. Can not test it.')
    def test_split_subject_par_c_binarize(self):
        subj_kfold = SubjectKFold(self.epoch_proc.use_game_par_c(), self.kfold_num, binarize_db=True)
        self._run_epoch_proc()
        ans = list()
        for train, test, val, subj in subj_kfold.split(1):
            ans.append((train, test))
        self._check_method(ans)

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Game_ParadigmC.DIR).exists(),
                         'Data for Game_ParadigmC does not exists. Can not test it.')
    def test_cross_split_par_c_binarized(self):
        subj_kfold = SubjectKFold(self.epoch_proc.use_game_par_c(), self.kfold_num, binarize_db=True)
        self._run_epoch_proc()
        ans = list()
        for train, test, val, subj in subj_kfold.split(cross_subject=True):
            ans.append((train, test))
        self._check_method(ans)


@unittest.skipUnless(Path(init_base_config('..')).joinpath(Game_ParadigmD.DIR).exists(),
                     'Data for Game_paradigmD does not exists. Can not test it.')
class TestDataHandler(unittest.TestCase):

    def test_big_data(self):
        path = Path(init_base_config('..'))
        subject = 1
        epoch_proc = OfflineDataPreprocessor(path)
        epoch_proc.use_game_par_d()
        feature_extraction = dict(
            feature_type=FeatureType.SPATIAL_FFT_POWER,
            fft_low=14, fft_high=30
        )
        epoch_proc.run(subject, **feature_extraction)
        file_list = epoch_proc.get_processed_db_source(subject, only_files=True)
        labels = epoch_proc.get_labels()
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        file_handler = DataHandler(file_list, label_encoder)
        dataset = file_handler.get_tf_dataset()
        from tensorflow import data as tf_data
        self.assertIsInstance(dataset, tf_data.Dataset)
        for d, l in dataset.take(5):
            print(d.numpy(), l.numpy())


if __name__ == '__main__':
    unittest.main()

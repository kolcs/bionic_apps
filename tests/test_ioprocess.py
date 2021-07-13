import unittest
from pathlib import Path
from time import time

from numpy import ndarray
from sklearn.preprocessing import LabelEncoder

from config import Game_ParadigmD
from preprocess import init_base_config, SubjectHandle, DataLoader, DataProcessor, \
    OfflineDataPreprocessor, OnlineDataPreprocessor, \
    FeatureType, SubjectKFold, load_pickle_data, DataHandler, Databases
from tests.utils import AVAILABLE_DBS, cleanup_fastload_data


class TestDataLoader(unittest.TestCase):

    def _test_get_subject_list(self):
        subj_list = self.loader.get_subject_list()
        self.assertIsInstance(subj_list, list)
        for subj in subj_list:
            self.assertIsInstance(subj, int)

    def _test_get_filenames(self):
        for subj in self.loader.get_subject_list():
            file_names = self.loader.get_filenames_for_subject(subj)
            self.assertIsInstance(file_names, list)
            self.assertTrue(all(Path(file).exists() for file in file_names))

    def _run_test(self, db_config_ver=-1):
        for db_name in AVAILABLE_DBS:
            with self.subTest(f'Database: {db_name.name}'):
                self.loader.use_db(db_name, db_config_ver)
                self.assertIsInstance(self.loader.get_subject_num(), int)
                self._test_get_subject_list()
                if db_config_ver == 0 and db_name in [Databases.PHYSIONET, Databases.BCI_COMP_IV_2A,
                                                      Databases.BCI_COMP_IV_2B, Databases.BCI_COMP_IV_1,
                                                      Databases.GIGA]:
                    with self.assertRaises(NotImplementedError):
                        self._test_get_filenames()
                else:
                    self._test_get_filenames()

    def test_no_defined_db(self):
        self.loader = DataLoader('..')
        self.assertRaises(AssertionError, self.loader.get_subject_num)

    def test_independent_days(self):
        self.loader = DataLoader('..', subject_handle=SubjectHandle.INDEPENDENT_DAYS)
        self._run_test()

    def test_independent_days_old_db_config(self):
        self.loader = DataLoader('..', subject_handle=SubjectHandle.INDEPENDENT_DAYS)
        self._run_test(db_config_ver=0)

    def test_mix_experiments(self):
        self.loader = DataLoader('..', subject_handle=SubjectHandle.MIX_EXPERIMENTS)
        self._run_test()

    def test_bci_comp(self):
        self.loader = DataLoader('..', subject_handle=SubjectHandle.BCI_COMP)
        for db_name in AVAILABLE_DBS:
            with self.subTest(f'Database: {db_name.name}'):
                self.loader.use_db(db_name)
                if db_name.name == 'BCI_COMP_IV_1':
                    self.assertRaises(ValueError, self.loader.get_subject_num)
                elif 'BCI_COMP' in db_name.name or 'GIGA' in db_name.name:
                    self.assertIsInstance(self.loader.get_subject_num(), int)
                    self._test_get_subject_list()
                    self._test_get_filenames()
                else:
                    self.assertRaises(ValueError, self.loader.get_subject_num)


class TestDataProcessor(unittest.TestCase):

    def test_no_process(self):
        loader = DataProcessor(base_config_path='..')
        self.assertEqual(len(loader.get_processed_subjects()), 0)

    def _check_method(self, subject_handle=SubjectHandle.INDEPENDENT_DAYS, db_config_ver=-1):
        data_proc = DataProcessor(base_config_path='..', subject_handle=subject_handle)
        for db_name in AVAILABLE_DBS:
            with self.subTest(f'Database: {db_name.name}'):
                data_proc.use_db(db_name, db_config_ver)
                subject = data_proc.get_subject_list()[0]
                self.assertRaises(NotImplementedError,
                                  data_proc.run, subject, FeatureType.AVG_FFT_POWER, fft_low=7, fft_high=14)

                if db_config_ver == 0 and db_name in [Databases.PHYSIONET, Databases.BCI_COMP_IV_1,
                                                      Databases.BCI_COMP_IV_2A, Databases.BCI_COMP_IV_2B,
                                                      Databases.GIGA]:
                    with self.assertRaises(NotImplementedError):
                        filenames = data_proc.get_filenames_for_subject(subject)
                else:
                    filenames = data_proc.get_filenames_for_subject(subject)
                    task_dict = data_proc._generate_db_from_file_list(str(filenames[0]))
                    self.assertIsInstance(task_dict, dict)
                    self.assertGreater(len(task_dict), 1)
                    win_ep = task_dict[list(task_dict)[0]]
                    self.assertIsInstance(win_ep, dict)
                    self.assertGreater(len(win_ep), 1)
                    feat_list = win_ep[list(win_ep)[0]]
                    self.assertIsInstance(feat_list, list)
                    self.assertGreater(len(feat_list), 0)
                    self.assertIsInstance(feat_list[0], tuple)
                    self.assertIsInstance(feat_list[0][0], ndarray)
                    self.assertIsInstance(feat_list[0][1], ndarray)

    def test_db_generation_independent_days(self):
        self._check_method()

    def test_db_generation_independent_days_old_db_config(self):
        self._check_method(db_config_ver=0)


class TestOfflinePreprocessorIndependent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cleanup_fastload_data()
        cls.epoch_proc = OfflineDataPreprocessor(base_config_path='..')
        cls.subj_ind = 0
        cls._old_conf_exp_error = None

    # def setUp(self):
    #     pass

    def _check_db(self, subj, **feature_extraction):
        self.epoch_proc.run(subject=subj, **feature_extraction)
        self.assertIsInstance(self.epoch_proc.get_processed_db_source(), dict)
        self.assertGreater(len(self.epoch_proc.get_processed_db_source()), 0)

    def _check_method(self, **feature_extraction):
        for db_name in AVAILABLE_DBS:
            with self.subTest(f'Database: {db_name.name}'):
                self.epoch_proc.use_db(db_name)
                subj = self.epoch_proc.get_subject_list()[self.subj_ind]
                self._check_db(subj, **feature_extraction)

    def test_avg_fft_pow(self):
        self._check_method(feature_type=FeatureType.AVG_FFT_POWER,
                           fft_low=14, fft_high=30)

    def test_avg_fft_pow_old_config(self):
        feature_extraction = dict(feature_type=FeatureType.AVG_FFT_POWER,
                                  fft_low=14, fft_high=30)
        for db_name in AVAILABLE_DBS:
            with self.subTest(f'Database: {db_name.name}'):
                self.epoch_proc.use_db(db_name, 0)

                if self._old_conf_exp_error is None:
                    subj = self.epoch_proc.get_subject_list()[:2]
                    self._check_db(subj, **feature_extraction)
                else:
                    with self.assertRaises(self._old_conf_exp_error):
                        subj = [2, 3]
                        self._check_db(subj, **feature_extraction)

    def test_fft_range(self):
        self._check_method(feature_type=FeatureType.FFT_RANGE,
                           fft_low=2, fft_high=30)

    def test_multi_avg_fft_pow(self):
        self._check_method(feature_type=FeatureType.MULTI_AVG_FFT_POW,
                           fft_ranges=[(7, 14), (14, 28), (28, 40)])

    def test_fast_load(self):
        tic = time()
        self.test_avg_fft_pow()
        self.assertLess(time() - tic, 1, 'Error in fastload...')

    @unittest.skipUnless(DataLoader('..').use_physionet().get_data_path().exists(),
                         'Data for Physionet does not exists. Can not test it.')
    def test_data_update(self):
        self.epoch_proc.use_physionet()
        subj = self.epoch_proc.get_subject_list()[self.subj_ind + 1]
        self._check_db(subj, feature_type=FeatureType.AVG_FFT_POWER,
                       fft_low=14, fft_high=30)
        self.assertGreater(len(self.epoch_proc.get_processed_db_source()), 1)


class TestOfflinePreprocessorMix(TestOfflinePreprocessorIndependent):

    @classmethod
    def setUpClass(cls):
        cleanup_fastload_data()
        cls.epoch_proc = OfflineDataPreprocessor(subject_handle=SubjectHandle.MIX_EXPERIMENTS,
                                                 base_config_path='..')
        cls.subj_ind = 0
        cls._old_conf_exp_error = NotImplementedError


class TestOfflinePreprocessorBciComp(TestOfflinePreprocessorIndependent):

    @classmethod
    def setUpClass(cls):
        cleanup_fastload_data()
        cls.epoch_proc = OfflineDataPreprocessor(subject_handle=SubjectHandle.BCI_COMP,
                                                 base_config_path='..')
        cls.subj_ind = 0
        cls._old_conf_exp_error = NotImplementedError

    def _check_method(self, **feature_extraction):
        for db_name in AVAILABLE_DBS:
            with self.subTest(f'Database: {db_name.name}'):
                self.epoch_proc.use_db(db_name)
                subj = 2
                if db_name.name == 'BCI_COMP_IV_1':
                    with self.assertRaises(ValueError):
                        subj = self.epoch_proc.get_subject_list()[self.subj_ind]
                elif 'BCI_COMP' in db_name.name or 'GIGA' in db_name.name:
                    subj = self.epoch_proc.get_subject_list()[self.subj_ind]
                else:
                    with self.assertRaises(ValueError):
                        subj = self.epoch_proc.get_subject_list()[self.subj_ind]

                if db_name.name == 'BCI_COMP_IV_1':
                    with self.assertRaises(ValueError):
                        self._check_db(subj, **feature_extraction)
                elif 'BCI_COMP' in db_name.name or 'GIGA' in db_name.name:
                    self._check_db(subj, **feature_extraction)
                else:
                    with self.assertRaises(ValueError):
                        self._check_db(subj, **feature_extraction)

    @unittest.skipUnless(DataLoader('..').use_bci_comp_4_2a().get_data_path().exists(),
                         'Data for BCI comp 4 2a does not exists. Can not test it.')
    def test_data_update(self):
        with self.assertRaises(ValueError):
            super(TestOfflinePreprocessorBciComp, self).test_data_update()

        self.epoch_proc.use_bci_comp_4_2a()
        subj = self.epoch_proc.get_subject_list()[self.subj_ind + 1]
        self._check_db(subj, feature_type=FeatureType.AVG_FFT_POWER,
                       fft_low=14, fft_high=30)
        self.assertGreater(len(self.epoch_proc.get_processed_db_source()), 1)


class TestOnlinePreprocessor(TestOfflinePreprocessorIndependent):

    @classmethod
    def setUpClass(cls):
        cleanup_fastload_data()
        cls.epoch_proc = OnlineDataPreprocessor(base_config_path='..', do_artefact_rejection=False)
        cls.subj_ind = 0
        cls._old_conf_exp_error = NotImplementedError

    def _check_method(self, **feature_extraction):
        for db_name in AVAILABLE_DBS:
            with self.subTest(f'Database: {db_name.name}'):
                self.epoch_proc.use_db(db_name)
                assert hasattr(self.epoch_proc._db_type, 'CONFIG_VER') and self.epoch_proc._db_type.CONFIG_VER > 0, \
                    f'Database generation test implemented for db with CONFIG_VER > 0'
                subj = self.epoch_proc.get_subject_list()[self.subj_ind]
                self._check_db(subj, **feature_extraction)


class TestOfflineSubjectKFold(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cleanup_fastload_data()
        cls.kfold_num = 5
        cls.epoch_proc = OfflineDataPreprocessor(base_config_path='..')

    def _run_epoch_proc(self, subj_num):
        feature_extraction = dict(
            feature_type=FeatureType.AVG_FFT_POWER,
            fft_low=14, fft_high=30
        )
        subject_list = self.epoch_proc.get_subject_list()[:subj_num]
        self.epoch_proc.run(subject=subject_list, **feature_extraction)

    def _check_db(self, ans, kfold_num=None):
        if kfold_num is None:
            kfold_num = self.kfold_num
        self.assertEqual(len(ans), kfold_num)

        for train, test in ans:
            self.assertTrue(all(Path(file).exists() for file in train))
            self.assertTrue(all(Path(file).exists() for file in test))

        window_list = load_pickle_data(ans[0][0][0])
        data = window_list[0]
        self.assertIsInstance(data[0], ndarray)
        self.assertIn(type(data[1]), [str, int, float, list, ndarray])

    @staticmethod
    def _distinct(list1, list2):
        return not any(item in list1 for item in list2)

    def _check_distinct_state(self, train, test, val=None):
        msg = 'Datasets can not contain the same data'
        self.assertTrue(self._distinct(train, test), msg)
        if val is not None:
            self.assertTrue(self._distinct(train, val), msg)
            self.assertTrue(self._distinct(val, test), msg)

    def _check_method(self, subj_ind=None, binarize=False, cross_subject=False, kfold_num=None,
                      validation_split=0.0):
        for db_name in AVAILABLE_DBS:
            with self.subTest(f'Database: {db_name.name}'):
                kfn = kfold_num
                self.epoch_proc.use_db(db_name)
                self._run_epoch_proc(subj_ind + 1 if not cross_subject else self.kfold_num)
                subj_kfold = SubjectKFold(self.epoch_proc, self.kfold_num,
                                          validation_split=validation_split, binarize_db=binarize)

                if type(subj_ind) is int:
                    subj = self.epoch_proc.get_subject_list()[subj_ind]
                    if not cross_subject and type(self.epoch_proc) is OnlineDataPreprocessor:
                        kfn = min(self.kfold_num, len(self.epoch_proc.get_filenames_for_subject(subj)))
                else:
                    subj = None

                ans = list()
                for train, test, val, subj in subj_kfold.split(subj, cross_subject=cross_subject):
                    self._check_distinct_state(train, test, val)
                    ans.append((train, test))

                if cross_subject and kfn is None:
                    kfn = min(self.kfold_num, len(self.epoch_proc.get_subject_list()))

                self._check_db(ans, kfn)

    def test_subject_split(self):
        self._check_method(subj_ind=0)

    def test_subject_split_binarized(self):
        self._check_method(subj_ind=0, binarize=True)

    def test_subject_split_validation(self):
        self._check_method(subj_ind=0, validation_split=.2)

    def test_subject_split_validation_binarized(self):
        self._check_method(subj_ind=0, validation_split=.2, binarize=True)

    def test_cross_subject_split(self):
        self._check_method(cross_subject=True)

    def test_cross_subject_split_one(self):
        self._check_method(subj_ind=0, cross_subject=True, kfold_num=1)

    def test_cross_subject_split_binarized(self):
        self._check_method(cross_subject=True, binarize=True)

    def test_cross_subject_split_one_validation(self):
        self._check_method(subj_ind=0, cross_subject=True, validation_split=.2, kfold_num=1)

    def test_cross_subject_split_validation_binarized_one(self):
        self._check_method(subj_ind=0, cross_subject=True, validation_split=.2, binarize=True, kfold_num=1)


class TestOnlineSubjectKFold(TestOfflineSubjectKFold):

    @classmethod
    def setUpClass(cls):
        cleanup_fastload_data()
        cls.kfold_num = 5
        cls.epoch_proc = OnlineDataPreprocessor(base_config_path='..', do_artefact_rejection=False)

    def test_cross_subject_split(self):
        pass

    def test_cross_subject_split_one(self):
        pass

    def test_cross_subject_split_binarized(self):
        pass

    def test_cross_subject_split_one_validation(self):
        pass

    def test_cross_subject_split_validation_binarized_one(self):
        pass


@unittest.skipUnless(Path(init_base_config('..')).joinpath(Game_ParadigmD().DIR).exists(),
                     'Data for Game_paradigmD does not exists. Can not test it.')
class TestDataHandler(unittest.TestCase):

    def test_big_data(self):
        subject_list = [1, 2]
        epoch_proc = OfflineDataPreprocessor(base_config_path='..')
        epoch_proc.use_game_par_d()
        feature_extraction = dict(
            feature_type=FeatureType.SPATIAL_AVG_FFT_POW,
            fft_low=14, fft_high=30
        )
        epoch_proc.run(subject_list, **feature_extraction)
        file_list = epoch_proc.get_processed_db_source(subject_list[0], only_files=True)
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

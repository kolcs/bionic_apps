import unittest
from pathlib import Path, WindowsPath, PosixPath
from shutil import rmtree
from time import time

from numpy import ndarray
from sklearn.preprocessing import LabelEncoder

from config import Physionet, Game_ParadigmC, Game_ParadigmD, DIR_FEATURE_DB
from preprocess import init_base_config, get_db_name_by_filename, SubjectHandle, DataLoader, DataProcessor, \
    OfflineDataPreprocessor, OnlineDataPreprocessor, \
    FeatureType, SubjectKFold, load_pickle_data, DataHandler


def get_available_databases():
    base_dir = Path(init_base_config('..'))
    avail_dbs = set()
    for file in base_dir.rglob('*'):
        try:
            avail_dbs.add(get_db_name_by_filename(file.as_posix()))
        except ValueError:
            pass
    return avail_dbs


available_databases = get_available_databases()


def cleanup_fastload_data():
    path = Path(init_base_config('..')).joinpath(DIR_FEATURE_DB)
    if path.exists():
        print('Removing old files. It may take longer...')
        rmtree(str(path))


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
            self.assertIn(type(file_names[0]), [str, WindowsPath, PosixPath])

    def _run_test(self):
        for db_name in available_databases:
            with self.subTest(f'Database: {db_name.name}'):
                self.loader.use_db(db_name)
                self.assertIsInstance(self.loader.get_subject_num(), int)
                self._test_get_subject_list()
                self._test_get_filenames()

    def test_no_defined_db(self):
        self.loader = DataLoader('..')
        self.assertRaises(AssertionError, self.loader.get_subject_num)

    def test_independent_days(self):
        self.loader = DataLoader('..', subject_handle=SubjectHandle.INDEPENDENT_DAYS)
        self._run_test()

    def test_mix_experiments(self):
        self.loader = DataLoader('..', subject_handle=SubjectHandle.MIX_EXPERIMENTS)
        self._run_test()

    def test_bci_comp(self):
        self.loader = DataLoader('..', subject_handle=SubjectHandle.BCI_COMP)
        for db_name in available_databases:
            with self.subTest(f'Database: {db_name.name}'):
                self.loader.use_db(db_name)
                if 'BCI_COMP' in db_name.name:
                    self.assertIsInstance(self.loader.get_subject_num(), int)
                    self._test_get_subject_list()
                    self._test_get_filenames()
                else:
                    self.assertRaises(ValueError, self.loader.get_subject_num)


class TestDataProcessor(unittest.TestCase):

    def test_no_process(self):
        loader = DataProcessor(base_config_path='..')
        self.assertEqual(len(loader.get_processed_subjects()), 0)

    def test_db_generation_days(self):
        data_proc = DataProcessor(base_config_path='..')
        for db_name in available_databases:
            with self.subTest(f'Database: {db_name.name}'):
                data_proc.use_db(db_name)
                assert hasattr(data_proc._db_type, 'CONFIG_VER') and data_proc._db_type.CONFIG_VER > 0, \
                    f'Database generation test implemented for db with CONFIG_VER > 0'
                subject = data_proc.get_subject_list()[0]
                self.assertRaises(NotImplementedError,
                                  data_proc.run, subject, FeatureType.AVG_FFT_POWER, fft_low=7, fft_high=14)
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


class TestOfflinePreprocessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cleanup_fastload_data()
        cls.epoch_proc = OfflineDataPreprocessor(base_config_path='..')
        cls.subj_ind = 0

    # def setUp(self):
    #     pass

    def _check_db(self, subj, **feature_extraction):
        self.epoch_proc.run(subject=subj, **feature_extraction)
        self.assertIsInstance(self.epoch_proc.get_processed_db_source(), dict)
        self.assertGreater(len(self.epoch_proc.get_processed_db_source()), 0)

    def _check_method(self, **feature_extraction):
        for db_name in available_databases:
            with self.subTest(f'Database: {db_name.name}'):
                self.epoch_proc.use_db(db_name)
                subj = self.epoch_proc.get_subject_list()[self.subj_ind]
                self._check_db(subj, **feature_extraction)

    def test_avg_fft_pow(self):
        self._check_method(feature_type=FeatureType.AVG_FFT_POWER,
                           fft_low=14, fft_high=30)

    def test_fft_range(self):
        self._check_method(feature_type=FeatureType.FFT_RANGE,
                           fft_low=14, fft_high=30)

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


class TestOnlinePreprocessor(TestOfflinePreprocessor):

    @classmethod
    def setUpClass(cls):
        cleanup_fastload_data()
        cls.epoch_proc = OnlineDataPreprocessor(base_config_path='..', do_artefact_rejection=False)
        cls.subj_ind = 0

    # def _check_db(self, subj, **feature_extraction):
    #     self.epoch_proc.run(subject=subj, **feature_extraction)
    #     self.assertIsInstance(self.epoch_proc.get_processed_db_source(), dict)
    #     self.assertGreater(len(self.epoch_proc.get_processed_db_source()), 0)

    def _check_method(self, **feature_extraction):
        for db_name in available_databases:
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

    # def setUp(self):
    #     pass

    def _run_epoch_proc(self):
        feature_extraction = dict(
            feature_type=FeatureType.AVG_FFT_POWER,
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

    @staticmethod
    def _distinct(list1, list2):
        return not any(item in list1 for item in list2)

    def _check_distinct_state(self, train, test, val=None):
        msg = 'Datasets can not contain the same data'
        self.assertTrue(self._distinct(train, test), msg)
        if val is not None:
            self.assertTrue(self._distinct(train, val), msg)
            self.assertTrue(self._distinct(val, test), msg)

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Physionet.DIR).exists(),
                         'Data for Physionet does not exists. Can not test it.')
    def test_split_subject_physio(self):
        subj_kfold = SubjectKFold(self.epoch_proc.use_physionet(), self.kfold_num)
        self._run_epoch_proc()
        ans = list()
        for train, test, val, subj in subj_kfold.split(1):
            self._check_distinct_state(train, test, val)
            ans.append((train, test))
        self._check_method(ans)

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Physionet.DIR).exists(),
                         'Data for Physionet does not exists. Can not test it.')
    def test_split_subject_physio_binarize(self):
        subj_kfold = SubjectKFold(self.epoch_proc.use_physionet(), self.kfold_num, binarize_db=True)
        self._run_epoch_proc()
        ans = list()
        for train, test, val, subj in subj_kfold.split(1):
            self._check_distinct_state(train, test, val)
            ans.append((train, test))
        self._check_method(ans)

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Physionet.DIR).exists(),
                         'Data for Physionet does not exists. Can not test it.')
    def test_cross_split_physio(self):
        subj_kfold = SubjectKFold(self.epoch_proc.use_physionet(), self.kfold_num)
        self._run_epoch_proc()
        ans = list()
        for train, test, val, subj in subj_kfold.split(cross_subject=True):
            self._check_distinct_state(train, test, val)
            ans.append((train, test))
        self._check_method(ans)

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Physionet.DIR).exists(),
                         'Data for Physionet does not exists. Can not test it.')
    def test_cross_split_one_physio(self):
        subj_kfold = SubjectKFold(self.epoch_proc.use_physionet(), self.kfold_num)
        self._run_epoch_proc()
        ans = list()
        for train, test, val, subj in subj_kfold.split(1, cross_subject=True):
            self._check_distinct_state(train, test, val)
            ans.append((train, test))
        self._check_method(ans, 1)

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Physionet.DIR).exists(),
                         'Data for Physionet does not exists. Can not test it.')
    def test_cross_split_one_validation_physio(self):
        subj_kfold = SubjectKFold(self.epoch_proc.use_physionet(), self.kfold_num,
                                  validation_split=.2)
        self._run_epoch_proc()
        ans = list()
        for train, test, val, subj in subj_kfold.split(1, cross_subject=True):
            self._check_distinct_state(train, test, val)
            ans.append((train, test))
        self._check_method(ans, 1)

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Physionet.DIR).exists(),
                         'Data for Physionet does not exists. Can not test it.')
    def test_cross_split_physio_binarized(self):
        subj_kfold = SubjectKFold(self.epoch_proc.use_physionet(), self.kfold_num, binarize_db=True)
        self._run_epoch_proc()
        ans = list()
        for train, test, val, subj in subj_kfold.split(cross_subject=True):
            self._check_distinct_state(train, test, val)
            ans.append((train, test))
        self._check_method(ans)

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Game_ParadigmC.DIR).exists(),
                         'Data for Game_ParadigmC does not exists. Can not test it.')
    def test_split_subject_par_c_binarize(self):
        subj_kfold = SubjectKFold(self.epoch_proc.use_game_par_c(), self.kfold_num, binarize_db=True)
        self._run_epoch_proc()
        ans = list()
        for train, test, val, subj in subj_kfold.split(1):
            self._check_distinct_state(train, test, val)
            ans.append((train, test))
        self._check_method(ans)

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Game_ParadigmC.DIR).exists(),
                         'Data for Game_ParadigmC does not exists. Can not test it.')
    def test_cross_split_par_c_binarized(self):
        subj_kfold = SubjectKFold(self.epoch_proc.use_game_par_c(), self.kfold_num, binarize_db=True)
        self._run_epoch_proc()
        ans = list()
        for train, test, val, subj in subj_kfold.split(cross_subject=True):
            self._check_distinct_state(train, test, val)
            ans.append((train, test))
        self._check_method(ans)


class TestOnlineSubjectKFold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cleanup_fastload_data()
        cls.kfold_num = 5
        cls.epoch_proc = OnlineDataPreprocessor(base_config_path='..')

    # def setUp(self):
    #     pass

    def _run_epoch_proc(self):
        feature_extraction = dict(
            feature_type=FeatureType.AVG_FFT_POWER,
            fft_low=14, fft_high=30
        )
        self.epoch_proc.run(subject=list(range(1, self.kfold_num + 1)), **feature_extraction)

    def _check_method(self, ans):
        self.assertGreater(len(ans), 1)

        train, test = ans[0]
        self.assertTrue(Path(train[0]).exists())
        self.assertTrue(Path(test[0]).exists())

        window_list = load_pickle_data(train[0])
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

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Physionet.DIR).exists(),
                         'Data for Physionet does not exists. Can not test it.')
    def test_split_subject_physio(self):
        subj_kfold = SubjectKFold(self.epoch_proc.use_physionet(), self.kfold_num)
        self._run_epoch_proc()
        ans = list()
        for train, test, val, subj in subj_kfold.split(1):
            self._check_distinct_state(train, test, val)
            ans.append((train, test))
        self._check_method(ans)

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Physionet.DIR).exists(),
                         'Data for Physionet does not exists. Can not test it.')
    def test_split_subject_physio_binarize(self):
        subj_kfold = SubjectKFold(self.epoch_proc.use_physionet(), self.kfold_num, binarize_db=True)
        self._run_epoch_proc()
        ans = list()
        for train, test, val, subj in subj_kfold.split(1):
            self._check_distinct_state(train, test, val)
            ans.append((train, test))
        self._check_method(ans)

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Physionet.DIR).exists(),
                         'Data for Physionet does not exists. Can not test it.')
    def test_cross_split_physio(self):
        subj_kfold = SubjectKFold(self.epoch_proc.use_physionet(), self.kfold_num)
        self._run_epoch_proc()

        with self.assertRaises(NotImplementedError, msg='Please define test for cross_subject=True case!'):
            ans = list()
            for train, test, val, subj in subj_kfold.split(cross_subject=True):
                self._check_distinct_state(train, test, val)
                ans.append((train, test))
            self._check_method(ans)

    @unittest.skipUnless(Path(init_base_config('..')).joinpath(Physionet.DIR).exists(),
                         'Data for Physionet does not exists. Can not test it.')
    def test_split_subject_validation_physio(self):
        subj_kfold = SubjectKFold(self.epoch_proc.use_physionet(), self.kfold_num,
                                  validation_split=.2)
        self._run_epoch_proc()
        ans = list()
        for train, test, val, subj in subj_kfold.split(1):
            self._check_distinct_state(train, test, val)
            ans.append((train, test))
        self._check_method(ans)


@unittest.skipUnless(Path(init_base_config('..')).joinpath(Game_ParadigmD.DIR).exists(),
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

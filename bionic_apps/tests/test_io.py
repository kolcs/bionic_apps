import unittest
from pathlib import Path

from bionic_apps.databases import EEG_Databases
from bionic_apps.preprocess.io import DataLoader, SubjectHandle
from bionic_apps.tests.utils import AVAILABLE_DBS


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
                if db_config_ver == 0 and db_name in [EEG_Databases.PHYSIONET, EEG_Databases.BCI_COMP_IV_2A,
                                                      EEG_Databases.BCI_COMP_IV_2B, EEG_Databases.BCI_COMP_IV_1,
                                                      EEG_Databases.GIGA]:
                    with self.assertRaises(NotImplementedError):
                        self._test_get_filenames()
                else:
                    self._test_get_filenames()

    def test_no_defined_db(self):
        self.loader = DataLoader()
        self.assertRaises(AssertionError, self.loader.get_subject_num)

    def test_independent_days(self):
        self.loader = DataLoader(subject_handle=SubjectHandle.INDEPENDENT_DAYS)
        self._run_test()

    def test_independent_days_old_db_config(self):
        self.loader = DataLoader(subject_handle=SubjectHandle.INDEPENDENT_DAYS)
        self._run_test(db_config_ver=0)

    def test_mix_experiments(self):
        self.loader = DataLoader(subject_handle=SubjectHandle.MIX_EXPERIMENTS)
        self._run_test()

    def test_bci_comp(self):
        self.loader = DataLoader(subject_handle=SubjectHandle.BCI_COMP)
        for db_name in AVAILABLE_DBS:
            with self.subTest(f'Database: {db_name.name}'):
                self.loader.use_db(db_name)
                if db_name in [EEG_Databases.BCI_COMP_IV_2A, EEG_Databases.BCI_COMP_IV_2B, EEG_Databases.GIGA]:
                    self.assertIsInstance(self.loader.get_subject_num(), int)
                    self._test_get_subject_list()
                    self._test_get_filenames()
                else:
                    self.assertRaises(ValueError, self.loader.get_subject_num)


# @unittest.skipUnless(Path(init_base_config('..')).joinpath(Game_ParadigmD().DIR).exists(),
#                      'Data for Game_paradigmD does not exists. Can not test it.')
# class TestDataHandler(unittest.TestCase):
#
#     def test_big_data(self):
#         subject_list = [1, 2]
#         epoch_proc = OfflineDataPreprocessor(base_config_path='..')
#         epoch_proc.use_game_par_d()
#         feature_extraction = dict(
#             feature_type=FeatureType.SPATIAL_AVG_FFT_POW,
#             fft_low=14, fft_high=30
#         )
#         epoch_proc.run(subject_list, **feature_extraction)
#         file_list = epoch_proc.get_processed_db_source(subject_list[0], only_files=True)
#         labels = epoch_proc.get_labels()
#         label_encoder = LabelEncoder()
#         label_encoder.fit(labels)
#         file_handler = DataHandler(file_list, label_encoder)
#         dataset = file_handler.get_tf_dataset()
#         from tensorflow import data as tf_data
#         self.assertIsInstance(dataset, tf_data.Dataset)
#         for d, l in dataset.take(5):
#             print(d.numpy(), l.numpy())


if __name__ == '__main__':
    unittest.main()

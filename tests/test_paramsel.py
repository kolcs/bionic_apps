import unittest
from pathlib import PurePath

from pandas import DataFrame, concat

from paramsel import _make_one_fft_test, _generate_table, parallel_search_for_fft_params
from preprocess import init_base_config, Databases


# todo: use individual data for test_generate_table instead of generated


class TestParameterSelection(unittest.TestCase):
    _res = None

    @classmethod
    def setUpClass(cls):
        cls._path = PurePath(init_base_config())
        cls._path = cls._path.joinpath('Game', 'paradigmD', 'subject2')

    def test_1_make_one_test(self):
        fft = [(i, i + 4) for i in range(20, 36, 4)]
        res = [
            _make_one_fft_test(str(self._path.joinpath('rec01.vhdr')), Databases.GAME_PAR_D, fft_low, fft_high)
            for fft_low, fft_high in fft]
        self.assertGreater(len(res), 0)
        self.assertIsInstance(res[0], DataFrame)
        TestParameterSelection._res = res

    def test_2_generate_table(self):
        res = concat(self._res)
        res = _generate_table(res[['FFT low', 'FFT high', 'Avg. Acc']],
                              ['FFT low', 'FFT high'], acc_diff=.1)
        self.assertFalse(res.empty, 'Result of _generate_table() can not be empty!')

    def test_3_parallel_search_for_fft_params(self):
        res = parallel_search_for_fft_params(str(self._path.joinpath('rec01.vhdr')), Databases.GAME_PAR_D,
                                             20, 40, 6, 5)
        self.assertGreater(len(res), 0)
        self.assertIsInstance(res, list)
        self.assertIsInstance(res[0], tuple)
        self.assertEqual(len(res[0]), 2)


if __name__ == '__main__':
    unittest.main()

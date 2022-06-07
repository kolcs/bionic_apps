import numpy as np
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin


def _get_fft(data, fs, method='pow', n=512):
    """Calculating the frequency power."""
    n_timeponts = data.shape[-1] if n is None else n
    fft_res = np.fft.rfft(data, n=n)
    if method == 'abs':
        fft_res = np.abs(fft_res)
    elif method == 'pow':
        fft_res = np.power(np.abs(fft_res), 2)  # todo: division is missing...
    else:
        raise NotImplementedError(f'{method} method is not defined in fft calculation.')
    freqs = np.fft.rfftfreq(n_timeponts, 1. / fs)
    return freqs, fft_res


def to_micro_volt(data):
    return data * 1e6


class FFTCalc(BaseEstimator, TransformerMixin):

    def __init__(self, fs, method='psd2', min_nfft=None, return_only_fft=False):
        self.fs = fs
        self.method = 'psd1' if method == 'psd' else method
        self.min_nfft = min_nfft
        self.return_only_fft = return_only_fft

    def fit(self, x, y=None):
        if self.min_nfft is not None:
            self.min_nfft = max(self.min_nfft, x.shape[-1])
        return self

    def transform(self, x, y=None):
        if 'fft' in self.method:
            freqs, fft_res = _get_fft(x, self.fs, self.method.strip('fft'), n=self.min_nfft)
        elif 'psd' in self.method:
            div = int(self.method.strip('psd'))
            freqs, fft_res = signal.welch(x, self.fs, nperseg=np.size(x, -1) // div,
                                          nfft=self.min_nfft)
        else:
            raise NotImplementedError(f'{self.method} is not implemented for FFT calculation.')

        if self.return_only_fft:
            return fft_res
        return freqs, fft_res


class AvgFFTCalc(BaseEstimator, TransformerMixin):

    def __init__(self, fft_low, fft_high):
        self.fft_low = fft_low
        self.fft_high = fft_high

    def fit(self, x, y=None):
        return self

    def transform(self, data, y=None):
        freqs, fft_res = data
        fft_width = self.fft_high - self.fft_low
        assert all(fft_width >= freqs[i] - freqs[i - 1]
                   for i in range(1, len(freqs))), \
            'Not enough feature points between {} and {} Hz'.format(self.fft_low, self.fft_high)
        fft_mask = (freqs >= self.fft_low) & (freqs <= self.fft_high)
        fft_power = np.average(fft_res[..., fft_mask], axis=-1)
        return fft_power


class MultiAvgFFTCalc(BaseEstimator, TransformerMixin):

    def __init__(self, fft_ranges, fs, method='psd2', nfft=None):
        self.fft_ranges = fft_ranges
        self.fs = fs
        self.method = method
        self.nfft = nfft

    def fit(self, x, y=None):
        return self

    def transform(self, data, y=None):
        if 'fft' in self.method:
            freqs, fft_res = _get_fft(data, self.fs, self.method.strip('fft'), n=self.nfft)
        elif 'psd' in self.method:
            div = int(self.method.strip('psd'))
            freqs, fft_res = signal.welch(data, self.fs, nperseg=np.size(data, -1) // div,
                                          nfft=self.nfft)
        else:
            raise NotImplementedError(f'{self.method} is not implemetned for FFT calculation.')

        data_list = list()
        for fft_low, fft_high in self.fft_ranges:
            fft_width = fft_high - fft_low
            assert all(fft_width >= freqs[i] - freqs[i - 1]
                       for i in range(1, len(freqs))), \
                'Not enough feature points between {} and {} Hz'.format(fft_low, fft_high)
            fft_mask = (freqs >= fft_low) & (freqs <= fft_high)
            fft_power = np.average(fft_res[..., fft_mask], axis=-1)
            data_list.append(fft_power)

        if len(data.shape) == 4:
            data = np.transpose(data_list, (1, 0, 2, 3))
        elif len(data.shape) == 3:
            data = np.transpose(data_list, (1, 0, 2))
        else:
            raise ValueError(f'Input data must be 3 or 4 dimensional. '
                             f'Got {data.shape} instead.')
        return data


def get_fft_ranges(feature_type, fft_low=None, fft_high=None,
                   fft_width=2, fft_step=2,
                   fft_ranges=None):
    if feature_type == 'avg_fft_pow':
        assert fft_low is not None and fft_high is not None, \
            'fft_low and fft_high must be defined for {} feature'.format(feature_type)
        fft_ranges = [(fft_low, fft_high)]

    elif feature_type == 'fft_range':
        assert fft_low is not None and fft_high is not None, \
            'fft_low and fft_high must be defined for {} feature'.format(feature_type)
        fft_ranges = [(f, f + fft_width) for f in np.arange(fft_low, fft_high, fft_step)
                      if f + fft_width <= fft_high]

    elif feature_type == 'multi_avg_fft_pow':
        assert type(fft_ranges) is list and type(fft_ranges[0]) is tuple, \
            'fft_ranges parameter not defined correctly for {} feature'.format(feature_type)
    else:
        raise ValueError(f'{feature_type} is not defined.')
    return fft_ranges


def get_avg_fft_transformer(feature_type, fs, fft_low=None, fft_high=None,
                            fft_method='psd2', fft_width=2, fft_step=2,
                            fft_ranges=None):
    fft_ranges = get_fft_ranges(feature_type, fft_low, fft_high, fft_width, fft_step, fft_ranges)
    return MultiAvgFFTCalc(fft_ranges, fs, fft_method)


from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one, make_pipeline
from joblib import Parallel, delayed


class FFTUnion(FeatureUnion):

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        y : array-like of shape (n_samples, n_outputs), default=None
            Targets for supervised learning.

        **fit_params : dict, default=None
            Parameters to pass to the fit method of the estimator.

        Returns
        -------
        X_t : array-like or sparse matrix of \
                shape (n_samples, sum_n_components)
            The `hstack` of results of transformers. `sum_n_components` is the
            sum of `n_components` (output dimension) over transformers.
        """
        results = self._parallel_func(X, y, fit_params, _fit_transform_one)
        if not results:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*results)
        self._update_transformer_list(transformers)

        return np.array(Xs).transpose((1, 0, 2))

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        Returns
        -------
        X_t : array-like or sparse matrix of \
                shape (n_samples, sum_n_components)
            The `hstack` of results of transformers. `sum_n_components` is the
            sum of `n_components` (output dimension) over transformers.
        """
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight)
            for name, trans, weight in self._iter()
        )
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        return np.array(Xs).transpose((1, 0, 2))


def get_multi_fft_transformer(fs, fft_ranges, *, method='psd2'):
    inner_ffts = [(f'unit{i}', make_pipeline(AvgFFTCalc(fft_low, fft_high)))
                  for i, (fft_low, fft_high) in enumerate(fft_ranges)]

    clf = make_pipeline(
        FFTCalc(fs, method),
        FFTUnion(inner_ffts, n_jobs=len(inner_ffts)) if len(inner_ffts) > 1 else inner_ffts[0][1]
    )

    return clf

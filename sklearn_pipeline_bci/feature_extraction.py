from enum import Enum, auto

import numpy as np
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


# features
class FeatureType(Enum):
    AVG_FFT_POWER = auto()
    FFT_RANGE = auto()
    MULTI_AVG_FFT_POW = auto()


def _get_fft(data, fs, method='pow', n=512):
    """Calculating the frequency power."""
    n_timeponts = data.shape[-1] if n is None else n
    fft_res = np.fft.rfft(data, n=n)
    if method == 'abs':
        fft_res = np.abs(fft_res)
    elif method == 'pow':
        fft_res = np.power(np.abs(fft_res), 2)
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
            fft_power = np.average(fft_res[:, :, :, fft_mask], axis=-1)
            data_list.append(fft_power)
        data = np.transpose(data_list, (1, 0, 2, 3))
        return data


def get_fft_ranges(feature_type, fft_low=None, fft_high=None,
                   fft_width=2, fft_step=2,
                   fft_ranges=None):
    if feature_type is FeatureType.AVG_FFT_POWER:
        assert fft_low is not None and fft_high is not None, \
            'fft_low and fft_high must be defined for {} feature'.format(feature_type.name)
        fft_ranges = [(fft_low, fft_high)]

    elif feature_type is FeatureType.FFT_RANGE:
        assert fft_low is not None and fft_high is not None, \
            'fft_low and fft_high must be defined for {} feature'.format(feature_type.name)
        fft_ranges = [(f, f + fft_width) for f in np.arange(fft_low, fft_high, fft_step)]

    elif feature_type is FeatureType.MULTI_AVG_FFT_POW:
        assert type(fft_ranges) is list and type(fft_ranges[0]) is tuple, \
            'fft_ranges parameter not defined correctly for {} feature'.format(feature_type.name)
    else:
        raise ValueError(f'{feature_type.name} is not defined.')
    return fft_ranges


def _get_avg_fft(feature_type, fs, fft_low=None, fft_high=None,
                 fft_method='psd2', fft_width=2, fft_step=2,
                 fft_ranges=None):
    fft_ranges = get_fft_ranges(feature_type, fft_low, fft_high, fft_width, fft_step, fft_ranges)
    return MultiAvgFFTCalc(fft_ranges, fs, fft_method)


def get_feature_extractor(feature_type, fs, scale=True, **kwargs):
    pipeline_steps = []
    if scale:
        pipeline_steps.append(FunctionTransformer(to_micro_volt))

    if feature_type in [FeatureType.AVG_FFT_POWER, FeatureType.FFT_RANGE, FeatureType.MULTI_AVG_FFT_POW]:
        pipeline_steps.append(_get_avg_fft(feature_type, fs, **kwargs))
    else:
        pass

    return make_pipeline(*pipeline_steps)

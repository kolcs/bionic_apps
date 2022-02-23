import mne
import numpy as np
from mne.decoding import CSP
# from pyriemann.spatialfilters import CSP
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn_pipeline_bci.utils import filter_mne_obj, window_epochs


class FBCSP(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=4):
        self.n_components = n_components
        self.fbcsp = []

    def _expand_and_check(self, x):
        if len(x.shape) == 3:
            x = np.expand_dims(x, axis=0)
        assert len(x.shape) == 4, 'Unsupported shape for FBCSP.'
        return x

    def fit(self, x, y):  # todo: (filt, ep, ch, time) ?
        x = self._expand_and_check(x)
        self.fbcsp = []
        for f_x in x:
            self.fbcsp.append(CSP(self.n_components, cov_est='epoch'
                                  ).fit(f_x, y))

        return self

    def transform(self, x):
        if len(self.fbcsp) == 0:
            raise RuntimeError('No filters available. Please first fit FBCSP '
                               'decomposition.')
        x = self._expand_and_check(x)

        csps = []
        for i, csp in enumerate(self.fbcsp):
            csps.append(csp.transform(x[i]))
        csps = np.array(csps).transpose((1, 0, 2))
        n, filt, comp = csps.shape
        csps = csps.reshape((n, filt * comp))
        return csps


class FilterBank:

    def __init__(self, f_low=4, f_high=40, f_step=4, f_width=4, n_jobs=1):
        self.filters = [(f, f + f_width) for f in range(f_low, f_high, f_step)
                        if f + f_width <= f_high]
        self.n_jobs = n_jobs

    def transform(self, x):
        filter_bank = [filter_mne_obj(x, l_freq=l_freq, h_freq=h_freq, n_jobs=self.n_jobs).get_data()
                       for l_freq, h_freq in self.filters]
        return np.array(filter_bank)


class WindowEpochs:

    def __init__(self, window_length, window_step, fs, shuffle=False):
        self.window_length = window_length
        self.window_step = window_step
        self.fs = fs
        self.shuffle = shuffle

    def _validate_data(self, data):
        if isinstance(data, np.ndarray):
            pass
        elif isinstance(data, (mne.Epochs, mne.EpochsArray)):
            data = data.get_data()
        else:
            raise TypeError(f'Expected types: mne.Epochs, mne.EpochsArray, ndarray. '
                            f'Received {type(data)}')
        return data

    def transform(self, x, y):
        if isinstance(x, list) or len(x.shape) == 4:  # FBCSP
            data = [window_epochs(self._validate_data(ep), self.window_length, self.window_step, self.fs)
                    for ep in x]  # np.hstack?
            windowed_data_shape = data[0].shape
            windowed_data = np.array([np.vstack(d) for d in data])
        else:
            data = window_epochs(self._validate_data(x),
                                 window_length=self.window_length,
                                 window_step=self.window_step, fs=self.fs)
            windowed_data_shape = data.shape
            windowed_data = np.vstack(data)

        groups = np.array([i // windowed_data_shape[1] for i in range(windowed_data_shape[0] * windowed_data_shape[1])])
        labels = np.array([y[i // windowed_data_shape[1]] for i in range(len(groups))])

        if self.shuffle:
            ind = np.arange(len(labels))
            np.random.shuffle(ind)
            labels = labels[ind]
            groups = groups[ind]

            if len(windowed_data.shape) == 3:
                windowed_data = windowed_data[ind]
            else:
                windowed_data = np.array([d[ind] for d in windowed_data])

        return windowed_data, labels, groups

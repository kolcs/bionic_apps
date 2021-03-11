from enum import Enum, auto

import numpy as np
from joblib import Parallel, delayed
from mne.viz import plot_topomap
from mne.viz.topomap import _prepare_topomap_plot
from scipy import stats


# features
class FeatureType(Enum):
    FFT_POWER = auto()
    FFT_RANGE = auto()
    MULTI_FFT_POWER = auto()
    SPATIAL_FFT_POWER = auto()
    SPATIAL_TEMPORAL = auto()
    RAW = auto()


def _get_fft_power(data, fs, get_abs=True):
    """Calculating the frequency power."""
    n_data_point, n_channel, n_timeponts = data.shape
    fft_res = np.fft.rfft(data)
    if get_abs:
        fft_res = np.abs(fft_res)
    # fft_res = fft_res**2
    freqs = np.fft.rfftfreq(n_timeponts, 1. / fs)
    return fft_res, freqs


def _init_interp(info, ch_type='eeg'):
    """spatial data creator initializer

    This function initialize the interpreter which can be used to generate spatially
    distributed data from eeg signal if it is None. Otherwise returns it.
    This is the simplest solution to initialize the coordinates.

    Parameters
    ----------
    info : mne.Info
        the interpreter
    ch_type : str

    Returns
    -------
    interp : mne.viz.topomap._GridData
        Interpreter for spatially distributed data creation, containing
        CloughTocher2DInterpolator.

    """
    data_picks, pos, merge_channels, names, ch_type, sphere, clip_origin = \
        _prepare_topomap_plot(info, ch_type, sphere=None)

    # IMPORTANT: remove [:2] from the end of the return in plot_topomap() function
    im, _, interp = plot_topomap(np.zeros(len(pos)), pos, show=False)
    return interp


def _calculate_spatial_interpolation(interp, data):
    """Spatial interpolation.

    This function mimics the mne.viz.plot_topomap() function, however it returns
    with real data instead of a matplotlib instance.

    Parameters
    ----------
    interp : mne.viz.topomap._GridData
        Interpolator for spatially distributed data creation, containing
        CloughTocher2DInterpolator.
    data : np.ndarray
        Data to interpolate. shape: (n,)

    Returns
    -------
    np.ndarray
        Spatially interpolated data.

    """
    assert interp is not None, 'Interpolator must be defined for spatal interpolation.'
    interp.set_values(data)
    spatial_data = interp()
    return spatial_data


def _crop_spatial_data(spatial_data):
    """Removing data from the border, creating ROUND electrode system.

    Parameters
    ----------
    spatial_data: numpy.array
    """
    r = np.size(spatial_data, axis=0) / 2
    for i in range(int(2 * r)):
        for j in range(int(2 * r)):
            if np.power(i - r, 2) + np.power(j - r, 2) > np.power(r, 2):
                spatial_data[i, j] = np.nan


# def _interpol_one_epoch_sp_temp(ep, interp, crop_img, i=None):
#     interp_list = list()
#     for t in range(ep.shape[-1]):
#         spatial_data = _calculate_spatial_interpolation(interp, ep[:, t])
#         if crop_img:
#             _crop_spatial_data(spatial_data)
#         spatial_data = stats.zscore(spatial_data, axis=None)
#         spatial_data[np.isnan(spatial_data)] = 0
#         interp_list.append(spatial_data)
#     interp_list = np.expand_dims(interp_list, axis=-1)  # shape: (time, width, height, color)
#     if i is None:
#         return interp_list
#     return i, interp_list


class FeatureExtractor:

    def __init__(self, feature_type, fs=None,
                 fft_low=None, fft_high=None, fft_step=2, fft_width=2, fft_ranges=None,
                 channel_list=None, info=None, crop=True):
        """Class for feature extraction.

        Parameters
        ----------
        feature_type : FeatureType
            Specify the features which will be created in the preprocessing phase.
        fs : int
            Sampling frequency.
        fft_low : float
            Low border for fft power calculation.
        fft_high : float
            High border for fft power calculation.
        fft_step : float
            Step number between fft ranges.
        fft_width : float
            The width between fft borders.
        fft_ranges : list of (float, float)
            a list of frequency ranges. Each each element of the list is a tuple, where the
            first element of a tuple corresponds to fft_min and the second is fft_max
        channel_list : list of int, optional
            Dummy eeg channel selection. Do not use it.
        info : mne.Info
            Info file for interpolator. It is required in case of Spatial interpolation.
        """
        self.feature_type = feature_type
        self.fs = fs

        self.fft_low = fft_low
        self.fft_high = fft_high
        self.fft_step = fft_step
        self.fft_width = fft_width
        self.fft_ranges = fft_ranges

        self.channel_list = channel_list
        self._crop = crop
        self._info = info  # todo: change back info to interp when it is possible...

        self._check_params()

    def _check_params(self):
        if self.feature_type == FeatureType.FFT_POWER:
            self._check_fft_low_and_high()
        elif self.feature_type == FeatureType.MULTI_FFT_POWER:
            self._check_fft_ranges()
        elif self.feature_type == FeatureType.FFT_RANGE:
            self._check_fft_low_and_high()
        elif self.feature_type == FeatureType.SPATIAL_FFT_POWER:
            if self.fft_ranges is not None:
                self._check_fft_ranges()
            else:
                self._check_fft_low_and_high()
            self._check_info()
        elif self.feature_type == FeatureType.SPATIAL_TEMPORAL:
            self._check_info()
        elif self.feature_type == FeatureType.RAW:
            pass  # no parameters are required for this feature

        else:
            raise NotImplementedError("Parameter constrains for {} are not defined.".format(self.feature_type.name))

    def _check_fft_low_and_high(self):
        assert self.fft_low is not None and self.fft_high is not None, \
            'fft_low and fft_high must be defined for {} feature'.format(self.feature_type.name)

    def _check_fft_ranges(self):
        assert type(self.fft_ranges) is list and type(self.fft_ranges[0]) is tuple, \
            'fft_ranges parameter not defined correctly for {} feature'.format(self.feature_type.name)

    def _check_info(self):
        assert self._info is not None, 'info must be defined {} feature'.format(self.feature_type.name)

    def calculate_multi_fft_power(self, data):
        """Calculating fft power in all ranges given in fft_ranges. (General case of FFT_RANGE)

        Parameters
        ----------
        data : np.ndarray
            eeg data shape: (data_points, n_channels, n_timeponts)

        Returns
        -------
        np.ndarray
            Feature extracted data. Shape: (data_points, n_fft, n_channels)

        """
        fft_res, freqs = _get_fft_power(data, self.fs)

        data = list()
        for fft_low, fft_high in self.fft_ranges:
            fft_mask = (freqs >= fft_low) & (freqs <= fft_high)
            assert np.any(fft_mask), 'Empty frequency range between {} and {} Hz'.format(fft_low, fft_high)
            fft_power = np.average(fft_res[:, :, fft_mask], axis=-1)
            data.append(fft_power)

        data = np.transpose(data, (1, 0, 2))
        return data

    def calculate_fft_power(self, data):
        """Calculating fft power for eeg data

        Parameters
        ----------
        data : np.ndarray
            eeg data shape: (n_datapoint, n_channels, n_timepoint)

        Returns
        -------
        np.ndarray
            Feature extracted data.
        """
        self.fft_ranges = [(self.fft_low, self.fft_high)]
        return self.calculate_multi_fft_power(data)

    def calculate_fft_range(self, data):
        """Calculating fft power all ranges between fft_low and fft_high.

        Parameters
        ----------
        data : np.ndarray
            eeg data shape: (data_points, n_channels, n_timeponts)

        Returns
        -------
        np.ndarray
            Feature extracted data.

        """
        self.fft_ranges = [(f, f + self.fft_width) for f in np.arange(self.fft_low, self.fft_high, self.fft_step)]
        return self.calculate_multi_fft_power(data)

    def calculate_spatial_fft_power(self, data, crop=True):
        """Spatial data from epochs.

        Creates spatially distributed data for each epoch in the window.

        Parameters
        ----------
        data : np.ndarray
            eeg data shape: (data_points, n_channels, n_timeponts)
        crop : bool
            To corp the values to 0 if its out of the eeg circle.

        Returns
        -------
        spatial_list : list of ndarray
            Data

        """
        if self.fft_ranges is None:
            self.fft_ranges = [(self.fft_low, self.fft_high)]
        epochs = self.calculate_multi_fft_power(data)

        def interpol_one_epoch(ep, info, crop_img, i=None):
            interp_list = list()
            interp = _init_interp(info)
            for fft_pow in ep:
                spatial_data = _calculate_spatial_interpolation(interp, fft_pow)
                if crop_img:
                    _crop_spatial_data(spatial_data)
                spatial_data[np.isnan(spatial_data)] = 0
                spatial_data *= (255.0 / spatial_data.max())  # scale to rgb image
                interp_list.append(spatial_data)
            interp_list = np.transpose(interp_list, (1, 2, 0))  # reformat to rgb image order
            if i is None:
                return interp_list
            return i, interp_list

        if len(epochs) > 1 and len(self.fft_ranges) > 1:
            res = dict(Parallel(n_jobs=-2)(
                delayed(interpol_one_epoch)(epochs[i], self._info, crop, i) for i in range(len(epochs))))
            res = [res[i] for i in range(len(res))]  # keep order!
        else:
            res = [interpol_one_epoch(ep, self._info, crop) for ep in epochs]
        return np.array(res)

    def calculate_spatial_temporal(self, data, crop=True):

        def interpol_one_epoch(ep, info, crop_img, i=None):
            interp_list = list()
            interp = _init_interp(info)
            for t in range(ep.shape[-1]):
                spatial_data = _calculate_spatial_interpolation(interp, ep[:, t])
                if crop_img:
                    _crop_spatial_data(spatial_data)
                spatial_data = stats.zscore(spatial_data, axis=None, nan_policy='omit')
                spatial_data[np.isnan(spatial_data)] = 0
                interp_list.append(spatial_data)
            interp_list = np.expand_dims(interp_list, axis=-1)  # shape: (time, width, height, color)
            if i is None:
                return interp_list
            return i, interp_list

        if len(data) > 1:
            res = dict(Parallel(n_jobs=-2)(
                delayed(interpol_one_epoch)(data[i], self._info, crop, i) for i in range(len(data))))
            res = [res[i] for i in range(len(res))]  # keep order!
        else:
            res = [interpol_one_epoch(ep, self._info, crop) for ep in data]
        return np.array(res)

    def run(self, data):

        if len(data.shape) == 2:
            data = np.expand_dims(data, 0)
        if self.channel_list is not None:
            data = data[:, self.channel_list, :]
            print('It is assumed that the reference electrode is POz!!!')

        if self.feature_type == FeatureType.FFT_POWER:
            feature = self.calculate_fft_power(data)
        elif self.feature_type == FeatureType.FFT_RANGE:
            feature = self.calculate_fft_range(data)
        elif self.feature_type == FeatureType.MULTI_FFT_POWER:
            feature = self.calculate_multi_fft_power(data)

        elif self.feature_type == FeatureType.SPATIAL_FFT_POWER:
            feature = self.calculate_spatial_fft_power(data, self._crop)
        elif self.feature_type == FeatureType.SPATIAL_TEMPORAL:
            feature = self.calculate_spatial_temporal(data, self._crop)
        elif self.feature_type == FeatureType.RAW:
            feature = data

        else:
            raise NotImplementedError('{} feature is not implemented'.format(self.feature_type))

        return feature

from enum import Enum, auto

import numpy as np


# features
class FeatureType(Enum):
    SPATIAL = auto()
    AVG_COLUMN = auto()
    COLUMN = auto()
    FFT_POWER = auto()
    FFT_RANGE = auto()
    MULTI_FFT_POWER = auto()
    SIMPLE_TIME_DOMAIN = auto()


def _get_fft_power(data, fs, get_abs=True):
    """Calculating the frequency power."""
    n_data_point, n_channel, n_timeponts = data.shape
    fft_res = np.fft.rfft(data)
    if get_abs:
        fft_res = np.abs(fft_res)
    # fft_res = fft_res**2
    freqs = np.fft.rfftfreq(n_timeponts, 1. / fs)
    return fft_res, freqs


def _init_interp(interp, epochs, ch_type='eeg'):
    """spatial data creator initializer

    This function initialize the interpreter which can be used to generate spatially
    distributed data from eeg signal if it is None. Otherwise returns it.
    This is the simplest solution to initialize the coordinates.

    Parameters
    ----------
    interp : None | mne.viz.topomap._GridData
        the interpreter
    epochs : mne.Epoch
    ch_type : str

    Returns
    -------
    interp : mne.viz.topomap._GridData
        Interpreter for spatially distributed data creation, containing
        CloughTocher2DInterpolator.

    """
    if interp is None:
        from mne.channels import read_layout, make_eeg_layout
        from mne.viz import plot_topomap
        from mne.viz.topomap import _prepare_topo_plot

        try:
            layout = make_eeg_layout(epochs.info)
        except RuntimeError:
            layout = read_layout('EEG1005')

        # ch_type = _get_ch_type(epochs, ch_type)
        picks, pos, merge_grads, names, ch_type = _prepare_topo_plot(
            epochs, ch_type, layout)
        data = epochs.get_data()[0, :, 0]

        # remove [:2] from the end of the return in plot_topomap()
        im, _, interp = plot_topomap(data, pos, show=False)

    return interp


def _calculate_spatial_interpolation(interp, data, crop=True):
    """Spatial interpolation.

    This function mimics the mne.viz.plot_topomap() function, however it returns
    with real data instead of a matplotlib instance.

    Parameters
    ----------
    interp : mne.viz.topomap._GridData
        Interpreter for spatially distributed data creation, containing
        CloughTocher2DInterpolator.
    data : ndarray
        Data to interpolate. shape: (n,)
    crop : bool
        If crop true values will be zeroed out outside of middle circle.

    Returns
    -------
    ndarray
        Spatially interpolated data.

    """
    interp.set_values(data)
    spatial_data = interp()

    # removing data from the border - ROUND electrode system
    if crop:
        r = np.size(spatial_data, axis=0) / 2
        for i in range(int(2 * r)):
            for j in range(int(2 * r)):
                if np.power(i - r, 2) + np.power(j - r, 2) > np.power(r, 2):
                    spatial_data[i, j] = 0
    return spatial_data


def calculate_spatial_data(interp, epochs, crop=True):
    """Spatial data from epochs.

    Creates spatially distributed data for each epoch in the window.

    Parameters
    ----------
    interp : mne.viz.topomap._GridData
        interpreter for spatially distributed data creation. Should be initialized first!
    epochs : mne.Epoch
        Data for spatially distributed data generation. Each epoch will have its own spatially data.
        The time points are averaged for each eeg channel.
    crop : bool
        To corp the values to 0 if its out of the eeg circle.

    Returns
    -------
    spatial_list : list of ndarray
        Data

    """
    interp = _init_interp(interp, epochs)
    spatial_list = list()
    data3d = epochs.get_data()

    for k in range(np.size(epochs, 0)):
        ep = data3d[k, :, :]
        # todo: psd --> average freq. range
        ep = np.average(ep, axis=1)  # average time for each channel
        spatial_data = _calculate_spatial_interpolation(interp, ep, crop)

        spatial_list.append(spatial_data)

    return spatial_list, interp


def _calculate_fft_power(data, fs, fft_low, fft_high, **kwargs):
    """Calculating fft power for eeg data

    Parameters
    ----------
    data : ndarray
        eeg data shape: (n_datapoint, n_channels, n_timepoint)
    fs : int
        Sampling frequency.
    fft_low : int
        Low border for fft power calculation.
    fft_high : int
        High border for fft power calculation.

    Returns
    -------
    ndarray
        Feature extracted data.
    """
    return _calculate_multi_fft_power(data, fs, [(fft_low, fft_high)])


def _calculate_fft_range(data, fs, fft_low, fft_high, fft_step=2, fft_width=2, **kwargs):
    """Calculating fft power all ranges between fft_low and fft_high.

    Parameters
    ----------
    data : ndarray
        eeg data shape: (data_points, n_channels, n_timeponts)
    fs : int
        Sampling frequency.
    fft_low : int
        Low border for fft power calculation.
    fft_high : int
        High border for fft power calculation.
    fft_step : int
        Step number between fft ranges.
    fft_width : int
        The width between fft borders.

    Returns
    -------
    ndarray
        Feature extracted data.

    """
    fft_ranges = [(f, f + fft_width) for f in np.arange(fft_low, fft_high, fft_step)]
    return _calculate_multi_fft_power(data, fs, fft_ranges)


def _calculate_multi_fft_power(data, fs, fft_ranges, **kwargs):
    """Calculating fft power in all ranges given in fft_ranges. (General case of FFT_RANGE)

        Parameters
        ----------
        data : ndarray
            eeg data shape: (data_points, n_channels, n_timeponts)
        fs : int
            Sampling frequency.
        fft_ranges : list of (float, float)
            a list of frequency ranges. Each each element of the list is a tuple, where the
            first element of a tuple corresponds to fft_min and the second is fft_max

        Returns
        -------
        ndarray
            Feature extracted data. Shape: (data_points, n_fft, n_channels)

    """
    assert type(fft_ranges) is list and type(fft_ranges[0]) is tuple, 'Invalid argument for MULTI_FFT_POWER'

    fft_res, freqs = _get_fft_power(data, fs)

    data = list()
    for fft_low, fft_high in fft_ranges:
        fft_mask = (freqs >= fft_low) & (freqs <= fft_high)
        assert np.any(fft_mask), 'Empty frequency range between {} and {} Hz'.format(fft_low, fft_high)
        fft_power = np.average(fft_res[:, :, fft_mask], axis=-1)
        data.append(fft_power)

    data = np.transpose(data, (1, 0, 2))
    return data


def _calculate_variance(data):
    n_data_point, n_channel, n_timeponts = data.shape
    n_variance = int(n_timeponts / 2)
    data = np.array([np.var(data[:, :, i - n_variance:i], axis=-1) for i in range(n_variance, n_timeponts)])
    data = np.transpose(data, (1, 2, 0))
    return data


def _calculate_time_domain_features(data, fs, **kwargs):
    return _calculate_variance(data)


def make_feature_extraction(feature_type, data, fs, *, channel_list=None, **feature_kwargs):
    """Feature extraction function.

    Makes the required feature extraction method.

    Parameters
    ----------
    feature_type : FeatureType
        Specify the features which will be created in the preprocessing phase.
    data : ndarray
        eeg data shape: (data_points, n_channels, n_timeponts) or (n_channels, n_timeponts)
    fs : int
        Sampling frequency.
    channel_list : list of int, optional
        Dummy eeg channel selection. Do not use it.
    feature_kwargs
        Arbitrary keyword arguments for feature extraction.

    Returns
    -------
    ndarray
        Feature extracted data.

    """
    if len(data.shape) == 2:
        data = np.expand_dims(data, 0)
    if channel_list is not None:
        data = data[:, channel_list, :]
        print('It is assumed that the reference electrode is POz!!!')

    if feature_type == FeatureType.AVG_COLUMN:
        data = np.average(data, axis=-1)
    elif feature_type == FeatureType.FFT_POWER:
        data = _calculate_fft_power(data, fs, **feature_kwargs)
    elif feature_type == FeatureType.FFT_RANGE:
        data = _calculate_fft_range(data, fs, **feature_kwargs)
    elif feature_type == FeatureType.MULTI_FFT_POWER:
        data = _calculate_multi_fft_power(data, fs, **feature_kwargs)

    elif feature_type == FeatureType.SIMPLE_TIME_DOMAIN:
        data = _calculate_time_domain_features(data, fs, **feature_kwargs)

    else:
        raise NotImplementedError('{} feature is not implemented'.format(feature_type))

    return data

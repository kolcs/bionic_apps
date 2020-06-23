from enum import Enum, auto

import numpy as np


# features
class Features(Enum):
    SPATIAL = auto()
    AVG_COLUMN = auto()
    COLUMN = auto()
    FFT_POWER = auto()
    FFT_RANGE = auto()
    MULTI_FFT_POWER = auto()
    # CHANNEL_SEL = auto()


def _init_interp(interp, epochs, ch_type='eeg'):
    """spatial data creator initializer

    This function initialize the interpreter which can be used to generate spatially distributed data
    from eeg signal if it is None. Otherwise returns it

    Parameters
    ----------
    interp : None | mne.viz.topomap._GridData
        the interpreter
    epochs : mne.Epoch
    ch_type : str

    Returns
    -------
    interp : mne.viz.topomap._GridData
        interpreter for spatially distributed data creation

    """
    if interp is None:
        from mne.channels import _get_ch_type, read_layout
        from mne.viz import plot_topomap
        from mne.viz.topomap import _prepare_topo_plot

        layout = read_layout('EEG1005')
        ch_type = _get_ch_type(epochs, ch_type)
        picks, pos, merge_grads, names, ch_type = _prepare_topo_plot(
            epochs, ch_type, layout)
        data = epochs.get_data()[0, :, 0]

        # remove [:2] from the end of the return in plot_topomap()
        im, _, interp = plot_topomap(data, pos, show=False)

    return interp


def calculate_spatial_data(interp, epochs, crop=True):
    """Spatial data from epochs

    Creates spatially distributed data for each epoch.

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

    for i in range(np.size(epochs, 0)):
        ep = data3d[i, :, :]
        ep = np.average(ep, axis=1)  # average time for each channel
        interp.set_values(ep)
        spatial_data = interp()

        # removing data from the border - ROUND electrode system
        if crop:
            r = np.size(spatial_data, axis=0) / 2
            for i in range(int(2 * r)):
                for j in range(int(2 * r)):
                    if np.power(i - r, 2) + np.power(j - r, 2) > np.power(r, 2):
                        spatial_data[i, j] = 0

        spatial_list.append(spatial_data)

    return spatial_list, interp


def _calculate_fft_power(data, fs, fft_low, fft_high):
    """Calculating fft power for eeg data

    Parameters
    ----------
    data : numpy.array
        eeg data shape: (n_datapoint, n_channels, n_timepoint)
    fs : int
        Sampling frequency.
    fft_low : int
        Low border for fft power calculation.
    fft_high : int
        High border for fft power calculation.

    Returns
    -------
    numpy.array
        Feature extracted data.
    """
    return _calculate_multi_fft_power(data, fs, [(fft_low, fft_high)])[:, 0, :]


def _calculate_fft_range(data, fs, fft_low, fft_high, fft_step=2, fft_width=2):
    """Calculating fft power all ranges between fft_low and fft_high.

    Parameters
    ----------
    data : numpy.array
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
    numpy.array
        Feature extracted data.

    """
    fft_ranges = [(f, f + fft_width) for f in np.arange(fft_low, fft_high, fft_step)]
    return _calculate_multi_fft_power(data, fs, fft_ranges)


def _calculate_multi_fft_power(data, fs, fft_ranges):
    """Calculating fft power in all ranges given in fft_ranges. (General case of FFT_RANGE)

        Parameters
        ----------
        data : numpy.array
            eeg data shape: (data_points, n_channels, n_timeponts)
        fs : int
            Sampling frequency.
        fft_ranges : list of tuple
            a list of frequency ranges. Each each element of the list is a tuple, where the
            first element corresponds to fft_min and the second is fft_max

        Returns
        -------
        numpy.array
            Feature extracted data. Shape: (data_points, n_fft, n_channels)

    """
    n_data_point, n_channel, n_timeponts = data.shape
    fft_res = np.abs(np.fft.rfft(data))
    # fft_res = fft_res**2
    freqs = np.linspace(0, fs / 2, int(n_timeponts / 2) + 1)

    data = list()

    for fft_low, fft_high in fft_ranges:
        ind = [i for i, f in enumerate(freqs) if fft_low <= f <= fft_high]
        assert len(ind) > 0, 'Empty frequency range between {} and {} Hz'.format(fft_low, fft_high)
        fft_power = np.average(fft_res[:, :, ind], axis=-1)
        data.append(fft_power)

    data = np.transpose(data, (1, 0, 2))
    return data


# def _channel_selection_test(data, fs, fft_ranges, channel):
#     if channel is None:
#         channel = 0
#     return _calculate_multi_fft_power(data, fs, fft_ranges)[:, :, channel]


def make_feature_extraction(feature, data, fs, fft_low=14, fft_high=30, fft_width=2, fft_step=2, channel_list=None):
    if len(data.shape) == 2:
        data = np.expand_dims(data, 0)
    if channel_list is not None:
        data = data[:, channel_list, :]
        print('It is assumed that the reference electrode is POz!!!')

    if feature == Features.AVG_COLUMN:
        data = np.average(data, axis=-1)
    elif feature == Features.FFT_POWER:
        data = _calculate_fft_power(data, fs, fft_low, fft_high)
    elif feature == Features.FFT_RANGE:
        data = _calculate_fft_range(data, fs, fft_low, fft_high, fft_step, fft_width)
    elif feature == Features.MULTI_FFT_POWER:
        assert type(fft_low) is list and type(fft_low[0]) is tuple, 'Invalid argument for MULTI_FFT_POWER'
        data = _calculate_multi_fft_power(data, fs, fft_low)

    # elif feature == Features.CHANNEL_SEL:
    #     data = _channel_selection_test(data, fs, fft_low, channel_list)

    else:
        raise NotImplementedError('{} feature is not implemented'.format(feature))

    return data

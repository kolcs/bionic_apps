import numpy as np


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


def calculate_fft_power(data, fs, fft_low, fft_high):
    """Calculating fft power for eeg data

    Parameters
    ----------
    data : numpy.array
        eeg data shape: (n_datapoint, n_channels, n_timepoint) or (n_channels, n_timepoint)
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
    if len(data.shape) == 2:
        data = np.expand_dims(data, 0)
    n = np.size(data, -1)
    fft_res = np.abs(np.fft.rfft(data))
    # fft_res = fft_res**2
    freqs = np.linspace(0, fs / 2, int(n / 2) + 1)
    ind = [i for i, f in enumerate(freqs) if fft_low <= f <= fft_high]
    data = np.average(fft_res[:, :, ind], axis=-1)
    return data


def calculate_fft_range(data, fs, fft_low, fft_high, fft_step=2, fft_width=2):
    """Calculating fft power all ranges between fft_low and fft_high.

    Parameters
    ----------
    data : numpy.array
        eeg data shape: (data_points, n_channels, n_timeponts) or (n_channels, n_timepoint)
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
    if len(data.shape) == 2:
        data = np.expand_dims(data, 0)

    n_data_point, n_channel, n_timeponts = data.shape
    fft_res = np.abs(np.fft.rfft(data))
    freqs = np.linspace(0, fs / 2, int(n_timeponts / 2) + 1)
    ind = [i for i, f in enumerate(freqs) if fft_low <= f <= fft_low + fft_width]
    data = np.average(fft_res[:, :, ind], axis=-1)
    n_fft = 1
    for flow in range(fft_low + fft_step, fft_high, fft_step):
        ind = [i for i, f in enumerate(freqs) if flow <= f <= flow + fft_width]
        fft_power = np.average(fft_res[:, :, ind], axis=-1)
        data = np.append(data, fft_power, axis=1)
        n_fft += 1
    data = data.reshape((n_data_point, n_fft, n_channel))  # checked
    return data

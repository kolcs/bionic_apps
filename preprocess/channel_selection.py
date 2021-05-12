from enum import Enum, auto

import numpy as np
from scipy import stats


class ChannelSelMode(Enum):
    COVARIANCE = auto()


def covariance_channel_selection(epochs, Ns=10, Ntr=10):
    """Correlation-based channel selection.

    Gives a list of the best correlated channels selected from a given epoch.
    Based on: https://doi.org/10.1016/j.neunet.2019.07.008

    Parameters
    ----------
    epochs : mne.Epochs
        Mne epochs, which the correlation will be calculated on.
    Ns : int
        Number of selected channels
    Ntr : int
        Number of trials

    Returns
    -------
    list
        List of the names of the selected channels

    """
    data = epochs.get_data()
    all_channel_names = np.asarray(epochs.ch_names)
    channel_count = np.zeros_like(epochs.ch_names, dtype=np.int)

    for epoch in data:
        z_score = stats.zscore(epoch, axis=-1)

        # channels -> N best correlate channel
        R = np.corrcoef(z_score)
        R_mean = R.mean(axis=0)

        # sort_index: original position in array (ascending order)
        sort_index = np.argsort(-R_mean)

        # Selecting the first Ns channel from an epoch
        channel_count[sort_index[:Ns]] += 1

    sort_index = np.argsort(-channel_count)
    selected_channels = all_channel_names[sort_index[:Ntr]]

    return selected_channels


class ChannelSelector:

    def __init__(self, channel_num=10, mode=ChannelSelMode.COVARIANCE):
        """Helper class for selecting different channel selection methods.

        This class stores the offline selected channel names for online use.

        Parameters
        ----------
        channel_num : int
            Number of the best EEG channels in the output.
        mode : ChannelSelMode
            Enum for selecting the mode of the EEG channel selection.
        """
        self._channel_num = channel_num
        self._selected_channels = list()
        self._mode = mode

    def offline_select(self, epochs):
        epochs = epochs.copy().pick('eeg')
        if self._mode == ChannelSelMode.COVARIANCE:
            self._selected_channels = covariance_channel_selection(epochs, self._channel_num, self._channel_num)
        else:
            raise NotImplementedError(f'{self._mode.name} channel selection mode is not implemented.')
        return self._selected_channels

    def online_select(self):
        assert len(self._selected_channels) > 0, 'Offline channel selection required.'
        return self._selected_channels

    @property
    def mode(self):
        return self._mode.name


if __name__ == '__main__':
    from preprocess import get_epochs_from_raw_with_gui

    ep = get_epochs_from_raw_with_gui()
    ch_sel = ChannelSelector(channel_num=10, mode=ChannelSelMode.COVARIANCE)
    print(f'\nSelected channels: {ch_sel.offline_select(ep)}')

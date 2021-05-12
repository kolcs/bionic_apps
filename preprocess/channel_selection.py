import numpy as np
from scipy import stats


def covariance_channel_selection(epochs, Ns=10, Ntr=10):
    """Correlation-based channel selection.

    Gives a list of the best correlated channels selected from a given epoch.

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


if __name__ == '__main__':
    from preprocess import get_epochs_from_raw_with_gui

    epochs = get_epochs_from_raw_with_gui()
    print(covariance_channel_selection(epochs))

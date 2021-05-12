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

    selected_channels = list()
    data = epochs.get_data()
    all_channel_names = epochs.ch_names

    channels_dict = {name: 0 for name in all_channel_names}

    for epoch in data:
        # for j, channel in enumerate(epoch):
        #     data[i][j] = stats.zscore(channel)
        z_score = stats.zscore(epoch, axis=-1)

        # channels -> N best correlate channel
        R = np.corrcoef(z_score)
        R_mean = R.mean(axis=0)

        # sort_index: original position in array (ascending order)
        sort_index = np.argsort(-1 * R_mean)

        # Selecting the first Ns channel from an epoch
        for Ns in range(Ns):
            channels_dict[all_channel_names[sort_index[Ns]]] = channels_dict[all_channel_names[sort_index[Ns]]] + 1

    ordered_dict = {k: v for k, v in sorted(channels_dict.items(), key=lambda item: item[1], reverse=True)}
    iterator = iter(ordered_dict.keys())
    for i in range(Ntr):
        selected_channels.append(next(iterator))

    return selected_channels


if __name__ == '__main__':
    from preprocess import get_epochs_from_raw_with_gui

    epochs = get_epochs_from_raw_with_gui()
    selected_channels = covariance_channel_selection(epochs)
    print(selected_channels)

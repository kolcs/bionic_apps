from pathlib import Path

import numpy as np
import pandas as pd
from mne.io import read_raw
from scipy import stats

from config import TTK_DB


def select_epochs():
    from preprocess import open_raw_with_gui, get_epochs_from_raw
    from config import Game_ParadigmD, TTK_DB
    raw = open_raw_with_gui()
    # print("raw:")
    # print(raw)
    # print("end")

    if "paradigmD" in ''.join(raw.filenames):
        task_dict = Game_ParadigmD.TRIGGER_TASK_CONVERTER
    elif "TTK" in ''.join(raw.filenames):
        task_dict = TTK_DB.TRIGGER_TASK_CONVERTER
    else:
        raise NotImplementedError

    epochs = get_epochs_from_raw(raw, task_dict, epoch_tmin=0, epoch_tmax=4, baseline=(None, .1))
    return epochs


def all_TTK_data():
    p = Path(r'C:\Users\Virág\Desktop\Önlab\database\TTK')
    l = list(p.rglob('*.vhdr'))
    return l


def table_from_selected_channels(N=10, Ns=10, Ntr=10):
    """ Creates a table from the selected channels

    Creates a list from the selected channels a writes it to a csv file. All rows show subjects with their selected
    channel list. The last row is the result of the final channel selection averaged from the subjects channel lists.

    Parameters
    ----------
    N : int
        Number of selected channels for the dataset
    Ns : int
        Number of selected channels for each epoch
    Ntr : int
        Number of trials for each epoch

    Returns
    -------
    pd.DataFrame
        Table with the selected channels

    """
    from preprocess import get_epochs_from_raw

    p = Path(r'C:\Users\Virág\Desktop\Önlab\database\TTK')
    l = list(p.rglob('*.vhdr'))
    # DROP_SUBJECTS = [01, 09, 17]

    raw = read_raw(str(l[0]))
    task_dict = TTK_DB.TRIGGER_TASK_CONVERTER
    epochs = get_epochs_from_raw(raw, task_dict, epoch_tmin=0, epoch_tmax=4, baseline=(None, .1))

    all_channel_names = epochs.ch_names
    all_channels_dict = {name: 0 for name in all_channel_names}  # ebben vannak az összesítettek

    table = {}

    final_selected_channels = list()

    for sample in l:
        # if sample != Path(r'C:\Users\Virág\Desktop\Önlab\database\TTK\subject09\rec02.vhdr'):
        # print(str(sample))
        # print(str(sample).find("subject01"))
        if not (str(sample).find("subject01") != -1 or str(sample).find("subject09") != -1 or str(sample).find(
                "subject17") != -1):
            # egy drop subject jó lenne ide
            raw = read_raw(sample)
            task_dict = TTK_DB.TRIGGER_TASK_CONVERTER
            epochs = get_epochs_from_raw(raw, task_dict, epoch_tmin=0, epoch_tmax=4, baseline=(None, .1))
            selected_channels = covariance_channel_sel(epochs, Ns, Ntr)
            # print(selected_channels)
            s = str(sample)
            table[s[s.find("TTK") + len("TTK") + 1:s.rfind("rec01") - 1]] = selected_channels
            for channel in selected_channels:
                all_channels_dict[channel] = all_channels_dict[channel] + 1

    ordered_dict = {k: v for k, v in sorted(all_channels_dict.items(), key=lambda item: item[1], reverse=True)}
    iterator = iter(ordered_dict.keys())
    for i in range(N):
        final_selected_channels.append(next(iterator))

    table["Final selected channels"] = final_selected_channels
    final_table = pd.DataFrame(table)
    final_table = final_table.transpose()
    print(final_table)
    final_table.to_csv(r'C:\Users\Virág\Desktop\Önlab\results.csv')

    return final_table


def covariance_channel_sel(epochs, Ns=10, Ntr=10):
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
    data = epochs.get_data()  # epochs*channels*time
    all_channel_names = epochs.ch_names

    channels_dict = {name: 0 for name in all_channel_names}

    for i, epoch in enumerate(data):
        for j, chanel in enumerate(epoch):
            data[i][j] = stats.zscore(chanel)

        R = np.corrcoef(epoch)
        R_mean = np.mean(R, axis=0)
        # channels -> N db legjobban korrelált channel

        sort_index = np.argsort(-1 * R_mean)
        # sort_index: hányadik indexen volt az eredeti arrayben a legjobban korrelált elem (csökkenő sorrendben)

        # Első Ns csatorna kiválasztása:
        for Ns in range(Ns):
            channels_dict[all_channel_names[sort_index[Ns]]] = channels_dict[all_channel_names[sort_index[Ns]]] + 1

    ordered_dict = {k: v for k, v in sorted(channels_dict.items(), key=lambda item: item[1], reverse=True)}
    iterator = iter(ordered_dict.keys())
    for i in range(Ntr):
        selected_channels.append(next(iterator))
    # print(selected_channels)
    return selected_channels


def covariance_channel_selection(N=10, Ns=10, Ntr=10):
    """Correlation-based channel selection

    Gives a list of the best correlated channels selected from the whole TTK dataset.
    For each epoch, it uses the covariance_channel_sel function, sums their results, and
    selects the best channels from them.

    Parameters
    ----------
    N : int
        Number of selected channels for the dataset
    Ns : int
        Number of selected channels for each epoch
    Ntr : int
        Number of trials for each epoch

    Returns
    -------
    list
        List of the names of the selected channels

    """

    from preprocess import get_epochs_from_raw

    p = Path(r'C:\Users\Virág\Desktop\Önlab\database\TTK')
    l = list(p.rglob('*.vhdr'))

    raw = read_raw(str(l[0]))
    task_dict = TTK_DB.TRIGGER_TASK_CONVERTER
    epochs = get_epochs_from_raw(raw, task_dict, epoch_tmin=0, epoch_tmax=4, baseline=(None, .1))

    all_channel_names = epochs.ch_names
    all_channels_dict = {name: 0 for name in all_channel_names}  # ebben vannak az összesítettek
    # print(all_channels_dict)

    final_selected_channels = list()

    # selected_channels = covariance_channel_sel(epochs)

    for sample in l:
        if sample != Path(r'C:\Users\Virág\Desktop\Önlab\database\TTK\subject09\rec02.vhdr'):
            raw = read_raw(sample)
            task_dict = TTK_DB.TRIGGER_TASK_CONVERTER
            epochs = get_epochs_from_raw(raw, task_dict, epoch_tmin=0, epoch_tmax=4, baseline=(None, .1))
            selected_channels = covariance_channel_sel(epochs, Ns, Ntr)
            # print(selected_channels)
            for channel in selected_channels:
                all_channels_dict[channel] = all_channels_dict[channel] + 1

    ordered_dict = {k: v for k, v in sorted(all_channels_dict.items(), key=lambda item: item[1], reverse=True)}
    iterator = iter(ordered_dict.keys())
    for i in range(N):
        final_selected_channels.append(next(iterator))

    print("End of channel selection:")
    print(final_selected_channels)

    return final_selected_channels


if __name__ == '__main__':
    # epochs = select_epochs()
    # selected_channels = covariance_channel_sel(epochs)
    covariance_channel_selection()
    # table_from_selected_channels()

import mne
import numpy as np


def use_mne_edf(filename):
    raw = mne.io.read_raw_edf(filename, preload=True)
    print(raw)
    print(raw.info)

    n = raw.info['nchan']  # trigger channel!
    signal_labels = raw.info['ch_names']
    fs = raw.info['sfreq']
    print(n, signal_labels, fs)
    data, time = raw[:n, :]
    print(np.shape(data))
    events = mne.find_events(raw)
    print(type(np.transpose(raw.find_edf_events())[0, 0]))

# TODO: https://mne-tools.github.io/stable/tutorials/philosophy.html
"""
- check info without preloading data
- check epoch -> create new database
- plot
- filter
- resample
https://mne-tools.github.io/stable/auto_tutorials/plot_info.html#tut-info-objects

"""

if __name__ == '__main__':
    pass

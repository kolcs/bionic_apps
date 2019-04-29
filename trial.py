import mne
import numpy as np


def use_mne_edf(filename):
    raw = mne.io.read_raw_edf(filename, preload=False, stim_channel='auto')
    # print(raw)
    print(raw.info)
    print(raw.find_edf_events())
    raw = raw.load_data()
    print(raw.find_edf_events())
    data, time = raw[:,:]
    print(data)

    # n = raw.info['nchan']  # trigger channel!
    # signal_labels = raw.info['ch_names']
    # fs = raw.info['sfreq']
    # print(n, signal_labels, fs)
    # data, time = raw[:n, :]
    # print(np.shape(data))
    # events = mne.find_events(raw)
    # print(type(np.transpose(raw.find_edf_events())[0, 0]))

    # sigbufs = np.zeros((n, f.getNSamples()[0]))
    # for i in range(n):
    #     sigbufs[i, :] = f.readSignal(i)  # channel number, start, number of points --> channel * data points
    # print(signal_labels)
    # print(f.readAnnotations())
    # print(np.shape(sigbufs))


if __name__ == '__main__':
    file = "/home/csabi/databases/physionet.org/physiobank/database/eegmmidb/S001/S001R03.edf"
    use_mne_edf(file)

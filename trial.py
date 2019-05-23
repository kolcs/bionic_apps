import mne
import numpy as np
from config import *
from preprocess.ioprocess import EEGFileHandler


def use_mne_edf(filename):
    raw = mne.io.read_raw_edf(filename, preload=False, stim_channel=None)
    # print(raw)
    print(raw.info)
    # print(raw.find_edf_events())
    # raw = raw.load_data()
    # print(raw.find_edf_events())
    # data, time = raw[:,:]
    # print(data)

    n = raw.info['nchan']  # trigger channel!
    signal_labels = raw.info['ch_names']
    fs = raw.info['sfreq']
    print(n, signal_labels, fs)
    ch_list = list()
    print(raw.pick_channels(ch_list))
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


def open_raw_file(filename, preload=True, stim_channel='auto'):
    ext = filename.split('.')[-1]

    switcher = {
        'edf': mne.io.read_raw_edf,
        'eeg': mne.io.read_raw_brainvision,  # todo: check...
    }
    # Get the function from switcher dictionary
    mne_io_read_raw = switcher.get(ext, lambda: "nothing")
    # Execute the function
    return mne_io_read_raw(filename, preload=preload, stim_channel=stim_channel)


def _get_concatenated_raw_file(filenames):
    raw_list = [open_raw_file(file) for file in filenames]
    raw = mne.io.concatenate_raws(raw_list)
    return raw


def mne_epochs_and_save(file):
    # raw = _get_concatenated_raw_file(files)
    raw = open_raw_file(file)
    # ann = list()
    # ann.append(raw.annotations.description)
    # ann.append(raw.annotations.onset * 160)
    events = mne.find_events(raw, shortest_event=0, stim_channel='STI 014', initial_event=True, consecutive=True)
    # print(ann)
    # print(events)
    event_id = {
        REST: 1,
        LEFT_HAND: 2,
        RIGHT_HAND: 3
    }
    tmin = 0
    tmax = 4
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        baseline=None, preload=True)

    for i in range(len(epochs)):
        print(epochs[i].events, epochs[i].tmax - epochs[i].tmin)  # todo: use it in filehandler!!!
    # epochs.plot(n_epochs=2, n_channels=65, block=True)
    # epochs.save('../tmp/data.fif', fmt='double')


def open_epoch(filename):
    epoch = mne.read_epochs(filename)
    epoch.plot(n_epochs=2, n_channels=65, block=True)
    print(epoch.info)
    print(epoch.events)
    print(epoch[REST].events)
    print(np.shape(epoch[REST].next()))


def test(file):
    handler = EEGFileHandler(file)
    print(handler.get_frequency())
    print(handler.get_channels())


if __name__ == '__main__':
    base_dir = "D:/BCI_stuff/databases/"  # MTA TTK
    # base_dir = 'D:/Csabi/'  # Home
    # base_dir = "D:/Users/Csabi/data"  # ITK
    # base_dir = "/home/csabi/databases/"  # linux
    files = [base_dir + "physionet.org/physiobank/database/eegmmidb/S001/S001R03.edf",
             base_dir + "physionet.org/physiobank/database/eegmmidb/S001/S001R04.edf"]
    # test(files[0])
    use_mne_edf(files[0])
    # mne_epochs_and_save(files[0])
    # open_epoch('../tmp/data.fif')

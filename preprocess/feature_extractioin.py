import mne
from preprocess.ioprocess import EEGFileHandler, OfflineEpochCreator, get_record_number, open_raw_file


def filter_on_file(filename, proc):
    raw = open_raw_file(filename)
    rec_num = get_record_number(filename)
    task_dict = proc.convert_task(rec_num)

    raw_alpha = raw.copy()
    raw_beta = raw.copy()
    raw_alpha = raw_alpha.filter(7, 13)
    raw_beta = raw_beta.filter(14, 30)

    events = mne.find_events(raw, shortest_event=0, stim_channel='STI 014', initial_event=True,
                             consecutive=True)
    epoch_alpha = mne.Epochs(raw_alpha, events, event_id=task_dict, tmin=0, tmax=4, preload=True)
    epoch_beta = mne.Epochs(raw_beta, events, event_id=task_dict, tmin=0, tmax=4, preload=True)

    print(raw.info['ch_names'])
    epoch_alpha['left hand'].plot(n_channels=len(raw.info['ch_names']) - 1, events=events, block=True)
    # epoch_alpha.average()
    # epoch_alpha.plot()
    # epoch_alpha.plot_psd_topomap()

    csp = mne.decoding.CSP(n_components=4)  # todo: continue!


if __name__ == '__main__':
    # base_dir = "D:/BCI_stuff/databases/"  # MTA TTK
    # base_dir = 'D:/Csabi/'  # Home
    base_dir = "D:/Users/Csabi/data/"  # ITK
    # base_dir = "/home/csabi/databases/"  # linux

    files = [base_dir + "physionet.org/physiobank/database/eegmmidb/S001/S001R03.edf",  # real hand
             base_dir + "physionet.org/physiobank/database/eegmmidb/S001/S001R04.edf",  # img hand
             base_dir + "physionet.org/physiobank/database/eegmmidb/S001/S001R05.edf",  # real hand-foot
             base_dir + "physionet.org/physiobank/database/eegmmidb/S001/S001R06.edf"]  # img hand-foot

    proc = OfflineEpochCreator(base_dir)
    proc.use_physionet()

    filter_on_file(files[0], proc)

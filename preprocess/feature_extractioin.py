import mne
import numpy as np
from matplotlib import pyplot as plt
from preprocess.ioprocess import EEGFileHandler, OfflineEpochCreator, get_record_number, open_raw_file


def plot_avg(epochs):
    ev_left = epochs['left hand'].average()
    ev_right = epochs['right hand'].average()
    ev_rest = epochs['rest'].average()

    f, axs = plt.subplots(1, 3, figsize=(10, 5))
    _ = f.suptitle('Left / Right hand', fontsize=20)
    _ = ev_left.plot(axes=axs[0], show=False, time_unit='s')
    _ = ev_right.plot(axes=axs[1], show=False, time_unit='s')
    _ = ev_rest.plot(axes=axs[2], show=False, time_unit='s')
    plt.tight_layout()


def plot_topo_psd(epochs, layout=None, bands=None, title=None, dB=True):
    if bands is None:
        bands = [(0, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 30, 'Beta'), (30, 45, 'Gamma')]
    if title is not None:
        bands = [(frm, to, name + ' ' + title) for frm, to, name in bands]

    epochs.plot_psd_topomap(bands=bands, layout=layout, dB=dB, show=False)


def plot_projs_topomap(epoch, ch_type='eeg', layout=None, title=None):
    fig = epoch.plot_projs_topomap(ch_type=ch_type, layout=layout)
    if title:
        fig.title(title)


def plot_csp(epoch, layout=None, n_components=4, title=None):
    labels = epoch.events[:, -1] - 1
    data = epoch.get_data()

    csp = mne.decoding.CSP(n_components=n_components)
    csp.fit_transform(data, labels)
    csp.plot_patterns(epoch.info, layout=layout, ch_type='eeg', show=False, title=title)


def artefact_correction(raw, l_freq=3, h_freq=40, order=5, layout=None):
    # sos = signal.iirfilter(order, (lowf, highf), btype='bandpass', ftype='butter', output='sos', fs=raw.info['sfreq'])
    iir_params = dict(order=order, ftype='butter', output='sos')
    raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir', iir_params=iir_params)

    raw.plot()
    raw.plot_psd()
    return

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    ica_method = 'fastica'
    n_components = 25
    decim = 3
    random_state = 12

    ica = mne.preprocessing.ICA(n_components=n_components, method=ica_method, random_state=random_state)

    reject = dict(mag=5e-12, grad=4000e-13, eeg=40e-6)  # todo: threshold and see the res in epochs, for sale: 20e-6

    ica.fit(raw, picks=picks, decim=decim, reject=reject)
    ica.plot_components(layout=layout)


def wavelet_time_freq(epochs, n_cycles=5, l_freq=7, h_freq=30, f_step=0.5, average=True, channels=None, title=None):
    freqs = np.arange(l_freq, h_freq, f_step)

    if channels is None:
        channels = epochs.info['ch_names']
        channels = channels[:-1]

    if average:
        power, itc = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=True, decim=3,
                                                   n_jobs=1)
    else:
        power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False, decim=3,
                                              n_jobs=1, average=False)

    indices = [power.ch_names.index(ch_name) for ch_name in channels]
    for i, ind in enumerate(indices):
        power.plot(picks=[ind], show=False, title=title + ' ' + channels[i])


def filter_on_file(filename, proc):
    raw = open_raw_file(filename)

    """MUST HAVE!!! Otherwise error!"""
    raw.rename_channels(lambda x: x.strip('.'))

    rec_num = get_record_number(filename)
    task_dict = proc.convert_task(rec_num)
    layout = mne.channels.read_layout('EEG1005')

    raw_alpha = raw.copy()
    raw_beta = raw.copy()
    raw_alpha = raw_alpha.filter(7, 13)
    raw_beta = raw_beta.filter(14, 30)

    events = mne.find_events(raw, shortest_event=0, stim_channel='STI 014', initial_event=True,
                             consecutive=True)
    epochs = mne.Epochs(raw, events, event_id=task_dict, tmin=0, tmax=4, preload=True)
    epoch_alpha = mne.Epochs(raw_alpha, events, event_id=task_dict, tmin=0, tmax=4, preload=True)
    epoch_beta = mne.Epochs(raw_beta, events, event_id=task_dict, tmin=0, tmax=4, preload=True)

    # n_ = min([len(epochs[task]) for task in task_dict])
    # for i in range(n_):
    #     for task in task_dict:
    #         ep = epochs[task]
    #         wavelet_time_freq(ep[i], channels=['Cz', 'C3', 'C4'], title=task+str(i))
    #         plot_topo_psd(ep[i], layout, title=task+str(i))
    #         # plot_projs_topomap(epochs[task], layout=layout, title=task)

    # epoch_alpha['left hand'].plot(n_channels=len(raw.info['ch_names']) - 1, events=events, block=True)

    # plot_avg(epoch_alpha)

    # # CSP
    # n_comp = 4
    # plot_csp(epoch_alpha, layout, n_components=n_comp, title='alpha range')
    # plot_csp(epoch_beta, layout, n_components=n_comp, title='beta range')

    artefact_correction(raw, layout=layout)  # todo: continue investigation

    plt.show()


if __name__ == '__main__':
    base_dir = "D:/BCI_stuff/databases/"  # MTA TTK
    # base_dir = 'D:/Csabi/'  # Home
    # base_dir = "D:/Users/Csabi/data/"  # ITK
    # base_dir = "/home/csabi/databases/"  # linux

    subj = 2
    rec = 8
    file = '{}physionet.org/physiobank/database/eegmmidb/S{:03d}/S{:03d}R{:02d}.edf'.format(base_dir, subj, subj, rec)

    proc = OfflineEpochCreator(base_dir)
    proc.use_physionet()

    filter_on_file(file, proc)

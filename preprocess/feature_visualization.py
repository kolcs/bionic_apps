import numpy as np

from preprocess import OfflineDataPreprocessor, FeatureType, Databases


def extract_features(feature, db_name,
                     subject_list=None,
                     epoch_tmin=0, epoch_tmax=4,
                     window_length=1, window_step=.1,
                     fast_load=True,
                     use_drop_subject_list=True,
                     filter_params=None):
    if filter_params is None:
        filter_params = {}

    proc = OfflineDataPreprocessor(
        epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
        window_length=window_length, window_step=window_step,
        fast_load=fast_load, filter_params=filter_params,
        use_drop_subject_list=use_drop_subject_list,
    )
    proc.use_db(db_name)
    if subject_list is None:
        subject_list = proc.get_subject_list()
    proc.run(subject_list, **feature)
    return proc


def show_mne_feature(proc, subject=None, task=None, epoch=0, window=0):
    assert proc.feature_type == FeatureType.RAW, 'This method is only compatible with RAW feature'
    if subject is None:
        subject = 1
    data, label = proc.get_feature(subject, task, epoch, window)
    label = label[0]
    epoch = proc.generate_mne_epoch(data)
    # now any kind of MNE feature extraction methods can be implemented and visualized

    # Example:
    # 1) feature generation. Can be done in feature_extraction.py also.
    from mne.time_frequency import tfr_morlet
    freqs = np.arange(2., 55.)
    n_cycles = freqs / 2.
    wavelet = tfr_morlet(epoch, freqs=freqs, n_cycles=n_cycles,
                         return_itc=False, average=True)
    # 2) feature visualization
    wavelet.plot(mode='mean',  # vmin=vmin, vmax=vmax,
                 title='Average of {} through all channels'.format(label), show=True)
    for ch in wavelet.ch_names:
        ch_wavelet = wavelet.copy().pick_channels([ch])
        ch_wavelet.plot(mode='mean',  # vmin=vmin, vmax=vmax,
                        title='{} : channel {}'.format(label, ch), show=True)


def show_numpy_feature(proc, subject=None, task=None, epoch=0, window=0):
    assert proc.feature_type == FeatureType.RAW, 'This method is only compatible with RAW feature'
    if subject is None:
        subject = 1
    data, label = proc.get_feature(subject, task, epoch, window)
    label = label[0]
    # now any kind of feature extraction methods can be implemented and visualized

    # Example:
    # 1) feature generation. Can be done in feature_extraction.py also.
    if len(data.shape) == 2:
        data = np.expand_dims(data, 0)
    from mne.time_frequency import tfr_array_morlet
    freqs = np.arange(2., 55.)
    n_cycles = freqs / 2.
    wavelet = tfr_array_morlet(data, proc.fs, freqs, n_cycles, use_fft=False, output='power')
    wavelet = wavelet[0, :, :, :]  # (n_epochs, n_chans, n_freqs, n_times)

    # 2) feature visualization
    from matplotlib import pyplot as plt
    for i, ch in enumerate(proc.info['ch_names']):
        plt.figure()
        plt.imshow(wavelet[i, :, :], origin='lower', extent=[0, np.size(wavelet, -1) / proc.fs, 0, np.size(wavelet, 1)],
                   aspect='auto')
        plt.title('{} : channel {}'.format(label, ch))
        plt.ylabel('frequencies')
        plt.xlabel('time')
        plt.show()


if __name__ == '__main__':
    db = Databases.PHYSIONET
    filter_par = dict(order=5, l_freq=1, h_freq=None)

    # mne feature visualization
    feature_extraction = dict(
        feature_type=FeatureType.RAW,
    )
    proc = extract_features(feature_extraction, db, subject_list=1, filter_params=filter_par)

    show_mne_feature(proc)
    show_numpy_feature(proc)

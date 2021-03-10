import numpy as np

from preprocess import OfflineDataPreprocessor, init_base_config, FeatureType, Databases


def extract_features(feature, db_name,
                     subject_list=None,
                     epoch_tmin=0, epoch_tmax=4,
                     window_length=1, window_step=.1,
                     use_drop_subject_list=True,
                     filter_params=None):
    if filter_params is None:
        filter_params = {}

    # generate database if not available
    proc = OfflineDataPreprocessor(
        init_base_config(),
        epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
        window_length=window_length, window_step=window_step,
        fast_load=False, filter_params=filter_params,
        use_drop_subject_list=use_drop_subject_list,
    )
    proc.use_db(db_name)
    if subject_list is None:
        subject_list = list(np.arange(proc.get_subject_num()) + 1)
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

    # example:
    from mne.time_frequency import tfr_morlet
    freqs = np.arange(2., 55.)
    n_cycles = freqs / 2.
    power = tfr_morlet(epoch, freqs=freqs, n_cycles=n_cycles,
                       return_itc=False, average=True)
    power.plot(mode='mean',  # vmin=vmin, vmax=vmax,
               title='Average of {} through all channels'.format(label), show=True)
    for ch in power.ch_names:
        p = power.copy().pick_channels([ch])
        p.plot(mode='mean',  # vmin=vmin, vmax=vmax,
               title='{} : channel {}'.format(label, ch), show=True)


if __name__ == '__main__':
    db = Databases.PHYSIONET
    filter_par = dict(order=5, l_freq=1, h_freq=None)

    # mne feature visualization
    feature_extraction = dict(
        feature_type=FeatureType.RAW,
        fft_low=7, fft_high=14
    )
    proc = extract_features(feature_extraction, db, subject_list=1, filter_params=filter_par)
    show_mne_feature(proc)

    # # numpy feature visualization
    # feature_extraction = dict(
    #     feature_type=FeatureType.FFT_POWER,
    #     fft_low=7, fft_high=14
    # )
    # proc = extract_features(feature_extraction, db)
    # show_feature(proc)

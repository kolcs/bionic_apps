import mne
import numpy as np

from .io import DataLoader, SubjectHandle, get_epochs_from_raw
from ..artifact_filtering.faster import ArtefactFilter
from ..databases import EEG_Databases
from ..feature_extraction import FeatureType, generate_features
from ..handlers.hdf5 import HDF5Dataset
from ..utils import standardize_eeg_channel_names, filter_mne_obj, balance_epoch_nums, _create_binary_label, \
    window_epochs


def generate_subject_data(files, loader, subj, filter_params,
                          epoch_tmin, epoch_tmax, window_length, window_step,
                          feature_type, feature_kwargs=None,
                          artifact_filter=None, balance_data=True,
                          binarize_labels=False):
    if feature_kwargs is None:
        feature_kwargs = {}

    task_dict = loader.get_task_dict()
    event_id = loader.get_event_id()
    print(f'\nSubject{subj}')
    raws = [mne.io.read_raw(file) for file in files]
    raw = mne.io.concatenate_raws(raws)
    raw.load_data()
    fs = raw.info['sfreq']

    standardize_eeg_channel_names(raw)
    try:  # check available channel positions
        mne.channels.make_eeg_layout(raw.info)
    except RuntimeError:  # if no channel positions are available create them from standard positions
        montage = mne.channels.make_standard_montage('standard_1005')  # 'standard_1020'
        raw.set_montage(montage, on_missing='warn')

    if len(filter_params) > 0:
        raw = filter_mne_obj(raw, **filter_params)

    epochs = get_epochs_from_raw(raw, task_dict,
                                 epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
                                 event_id=event_id)
    del raw

    if artifact_filter is not None:
        epochs = artifact_filter.offline_filter(epochs)

    ep_labels = [list(epochs[i].event_id)[0] for i in range(len(epochs))]

    if balance_data:
        epochs, ep_labels = balance_epoch_nums(epochs, ep_labels)

    if binarize_labels:
        ep_labels = [_create_binary_label(label) for label in ep_labels]

    # window the epochs
    windowed_data = window_epochs(epochs.get_data(),
                                  window_length=window_length, window_step=window_step,
                                  fs=fs)
    info = epochs.info
    del epochs

    groups = [i // windowed_data.shape[1] for i in range(windowed_data.shape[0] * windowed_data.shape[1])]
    labels = [ep_labels[i // windowed_data.shape[1]] for i in range(len(groups))]
    windowed_data = np.vstack(windowed_data)

    windowed_data = generate_features(windowed_data, fs, feature_type, info=info, **feature_kwargs)
    return windowed_data, labels, [subj] * len(labels), groups, fs


def generate_eeg_db(db_name, db_filename, feature_type=FeatureType.RAW,
                    epoch_tmin=0, epoch_tmax=4,
                    window_length=2, window_step=.1,
                    feature_kwargs=None,
                    use_drop_subject_list=True,
                    filter_params=None,
                    do_artefact_rejection=True,
                    balance_data=True,
                    subject_handle=SubjectHandle.INDEPENDENT_DAYS,
                    base_dir='.', fast_load=True,
                    n_subjects='all'):
    if filter_params is None:
        filter_params = {}
    if feature_kwargs is None:
        feature_kwargs = {}

    # removing data which does not affect fast_load
    feature_params = locals().copy()
    feature_params.pop('db_filename')
    feature_params.pop('fast_load')
    feature_params.pop('base_dir')

    loader = DataLoader(use_drop_subject_list=use_drop_subject_list,
                        subject_handle=subject_handle,
                        base_config_path=base_dir)
    loader.use_db(db_name)

    database = HDF5Dataset(db_filename, feature_params)

    if not (database.exists() and fast_load):
        if n_subjects == 'all':
            subject_list = loader.get_subject_list()
        else:
            assert isinstance(n_subjects, int), 'n_subject must be an integer'
            subject_list = loader.get_subject_list()[:n_subjects]

        for subj in subject_list:
            files = loader.get_filenames_for_subject(subj)
            artifact_filter = ArtefactFilter(apply_frequency_filter=False) if do_artefact_rejection else None
            subj_data = generate_subject_data(
                files, loader, subj, filter_params,
                epoch_tmin, epoch_tmax, window_length, window_step,
                feature_type, feature_kwargs,
                artifact_filter, balance_data,
                binarize_labels=db_name is EEG_Databases.GAME_PAR_D
            )
            database.add_data(*subj_data)
        database.close()

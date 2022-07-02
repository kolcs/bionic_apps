from pathlib import Path
from time import time

import mne
import numpy as np
from joblib import Parallel, delayed
from psutil import cpu_count

from .io import DataLoader, SubjectHandle, get_epochs_from_raw
from ..artifact_filtering.faster import ArtefactFilter
from ..databases import EEG_Databases
from ..feature_extraction import FeatureType, generate_features
from ..handlers.hdf5 import HDF5Dataset
from ..utils import standardize_eeg_channel_names, filter_mne_obj, balance_epoch_nums, _create_binary_label, \
    window_epochs

CPU_THRESHOLD = 10


def generate_subject_data(files, loader, subj, filter_params,
                          epoch_tmin, epoch_tmax, window_length, window_step,
                          artifact_filter=None, balance_data=True,
                          binarize_labels=False):
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

    from bionic_apps.preprocess.data_augmentation import do_augmentation
    ep_data, ep_labels, ep_ind, orig_ind = do_augmentation(epochs.get_data(), ep_labels)

    info = epochs.info
    del epochs

    # window the epochs
    windowed_data = window_epochs(ep_data,
                                  window_length=window_length, window_step=window_step,
                                  fs=fs)

    num = windowed_data.shape[0] * windowed_data.shape[1]
    groups = [ep_ind[i // windowed_data.shape[1]] for i in range(num)]
    labels = [ep_labels[i // windowed_data.shape[1]] for i in range(num)]
    orig_ind = [orig_ind[i // windowed_data.shape[1]] for i in range(num)]
    windowed_data = np.vstack(windowed_data)
    return windowed_data, labels, [subj] * len(labels), groups, orig_ind, fs, info


def _save_one_subject_data(feature_type, feature_kwargs, db_loader, subj,
                           epoch_tmin, epoch_tmax, window_length, window_step,
                           filter_params, balance_data, binarize_labels,
                           do_artefact_rejection, db_path):
    db_filename = db_path.joinpath(f'subject{subj}_db.hdf5')
    db_filename.unlink(missing_ok=True)
    database = HDF5Dataset(db_filename)
    files = db_loader.get_filenames_for_subject(subj)
    artifact_filter = ArtefactFilter(apply_frequency_filter=False) if do_artefact_rejection else None
    windowed_data, labels, subj_ind, ep_ind, fs, info = generate_subject_data(
        files, db_loader, subj, filter_params,
        epoch_tmin, epoch_tmax, window_length, window_step,
        artifact_filter, balance_data,
        binarize_labels=binarize_labels
    )
    windowed_data = generate_features(windowed_data, fs, feature_type, info=info, **feature_kwargs)
    database.add_data(windowed_data, labels, subj_ind, ep_ind, fs)
    database.close()
    return db_filename


def _merge_database(base_db, subject_files):
    for i, file in enumerate(subject_files):
        print(f'\rMerging subject databases: {i * 100. / len(subject_files):.2f}%', end='')
        subj_db = HDF5Dataset(file)
        labels = subj_db.get_y()
        subj_ind = subj_db.get_subject_group()
        ep_ind = subj_db.get_epoch_group()
        fs = subj_db.get_fs()
        windowed_data = subj_db.get_data(np.arange(len(labels)))
        base_db.add_data(windowed_data, labels, subj_ind, ep_ind, fs)
        subj_db.close()
        Path(file).unlink()
    print('\rMerging subject databases: 100.00%')


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
                    n_subjects='all', n_jobs=-2):
    if filter_params is None:
        filter_params = {}
    if feature_kwargs is None:
        feature_kwargs = {}

    # removing data which does not affect fast_load
    feature_params = locals().copy()
    feature_params.pop('db_filename')
    feature_params.pop('fast_load')
    feature_params.pop('base_dir')
    feature_params.pop('n_jobs')

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

        tic = time()
        if cpu_count() < CPU_THRESHOLD:  # parallel db generation is slow if there is not enough cpu_cores
            for subj in subject_list:
                files = loader.get_filenames_for_subject(subj)
                artifact_filter = ArtefactFilter(apply_frequency_filter=False) if do_artefact_rejection else None
                windowed_data, labels, subj_ind, ep_ind, fs, info = generate_subject_data(
                    files, loader, subj, filter_params,
                    epoch_tmin, epoch_tmax, window_length, window_step,
                    artifact_filter, balance_data,
                    binarize_labels=db_name is EEG_Databases.GAME_PAR_D
                )
                windowed_data = generate_features(windowed_data, fs, feature_type, info=info, **feature_kwargs)
                database.add_data(windowed_data, labels, subj_ind, ep_ind, fs)
        else:
            subj_db_files = Parallel(n_jobs)(
                delayed(_save_one_subject_data)(feature_type, feature_kwargs, loader, subj,
                                                epoch_tmin, epoch_tmax, window_length, window_step,
                                                filter_params, balance_data,
                                                db_name is EEG_Databases.GAME_PAR_D,
                                                do_artefact_rejection,
                                                db_filename.parent) for subj in subject_list)
            _merge_database(database, subj_db_files)

        database.close()
        print(f'DB generated under {(time() - tic) / 60:.2f} minutes')

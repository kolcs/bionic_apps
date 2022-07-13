from pathlib import Path
from time import time

import mne
import numpy as np
from joblib import Parallel, delayed
from psutil import cpu_count

from .data_augmentation import do_augmentation
from .io import DataLoader, SubjectHandle, get_epochs_from_raw, get_epochs_from_raw_annot
from ..artifact_filtering.faster import ArtefactFilter
from ..databases import Databases
from ..databases.coreg_mindrove import MindRoveCoreg
from ..feature_extraction import FeatureType, generate_features
from ..handlers.hdf5 import HDF5Dataset
from ..utils import standardize_eeg_channel_names, filter_mne_obj, balance_epoch_nums, _create_binary_label, \
    window_data

CPU_THRESHOLD = 10


def generate_subject_data(files, loader, subj, filter_params,
                          epoch_tmin, epoch_tmax, window_length, window_step,
                          artifact_filter=None, balance_data=True,
                          binarize_labels=False, augment_data=False,
                          ch_selection=None):
    print(f'\nSubject{subj}')
    raws = [mne.io.read_raw(file) for file in files]
    raw = mne.io.concatenate_raws(raws)
    raw.load_data()
    raw = raw.pick(ch_selection)
    fs = raw.info['sfreq']

    if ch_selection != 'emg':
        standardize_eeg_channel_names(raw)
        try:  # check available channel positions
            mne.channels.make_eeg_layout(raw.info)
        except RuntimeError:  # if no channel positions are available create them from standard positions
            montage = mne.channels.make_standard_montage('standard_1005')  # 'standard_1020'
            raw.set_montage(montage, on_missing='warn')

    if filter_params is not None:
        if ch_selection == 'all':
            pick_list = ['eeg', 'emg', 'eog']
            for pick, fpars in filter_params.items():
                assert pick in pick_list, f'filter_params are not defined well. ' \
                                          f'Keys must be in {pick_list}. Got {filter_params}'
                raw = filter_mne_obj(raw, picks=pick, **fpars)
        else:
            raw = filter_mne_obj(raw, picks=ch_selection, **filter_params)

    if isinstance(loader._db_type, MindRoveCoreg):
        ep_data, ep_labels, ep_min, _ = get_epochs_from_raw_annot(raw, return_min_max=True)
        assert window_length <= ep_min, f'The shortest epoch is {ep_min:.4f} sec long. ' \
                                        f'Can not make {window_length} sec long windows.'
    else:
        task_dict = loader.get_task_dict()
        event_id = loader.get_event_id()
        epochs = get_epochs_from_raw(raw, task_dict,
                                     epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
                                     event_id=event_id)

        if artifact_filter is not None:
            epochs = artifact_filter.offline_filter(epochs)

        ep_labels = [list(epochs[i].event_id)[0] for i in range(len(epochs))]

        if balance_data:
            epochs, ep_labels = balance_epoch_nums(epochs, ep_labels)

        if binarize_labels:
            ep_labels = [_create_binary_label(label) for label in ep_labels]

        ep_data = epochs.get_data()
        del epochs

    if augment_data:
        ep_data, ep_labels, ep_ind, ep_orig_mask = do_augmentation(ep_data, ep_labels)
    else:
        ep_ind = np.arange(len(ep_labels))
        ep_orig_mask = [True] * len(ep_labels)

    info = raw.info
    del raw

    # window the epochs
    windowed_data, groups, labels, orig_mask = [], [], [], []
    for i, ep in enumerate(ep_data):
        window = window_data(ep, window_length, window_step, fs)
        n_windows = window.shape[0]
        windowed_data.append(window)
        groups.extend([ep_ind[i]] * n_windows)
        labels.extend([ep_labels[i]] * n_windows)
        orig_mask.extend([ep_orig_mask[i]] * n_windows)

    windowed_data = np.vstack(windowed_data)
    del ep_data

    return windowed_data, labels, [subj] * len(labels), groups, orig_mask, fs, info


def _save_one_subject_data(feature_type, feature_kwargs, db_loader, subj,
                           epoch_tmin, epoch_tmax, window_length, window_step,
                           filter_params, balance_data, binarize_labels,
                           do_artefact_rejection, db_path,
                           augment_data, ch_selection):
    db_filename = db_path.joinpath(f'subject{subj}_db.hdf5')
    db_filename.unlink(missing_ok=True)
    database = HDF5Dataset(db_filename)
    files = db_loader.get_filenames_for_subject(subj)
    artifact_filter = ArtefactFilter(apply_frequency_filter=False) if do_artefact_rejection else None
    windowed_data, labels, subj_ind, ep_ind, orig_mask, fs, info = generate_subject_data(
        files, db_loader, subj, filter_params,
        epoch_tmin, epoch_tmax, window_length, window_step,
        artifact_filter, balance_data,
        binarize_labels=binarize_labels,
        augment_data=augment_data,
        ch_selection=ch_selection
    )
    windowed_data = generate_features(windowed_data, fs, feature_type, info=info, **feature_kwargs)
    database.add_data(windowed_data, labels, subj_ind, ep_ind, orig_mask, fs)
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
        orig_mask = subj_db.get_orig_mask()
        windowed_data = subj_db.get_data(np.arange(len(labels)))
        base_db.add_data(windowed_data, labels, subj_ind, ep_ind, orig_mask, fs)
        subj_db.close()
        Path(file).unlink()
    print('\rMerging subject databases: 100.00%')


def generate_db(db_name, db_filename, feature_type=FeatureType.RAW,
                epoch_tmin=0, epoch_tmax=4,
                window_length=2, window_step=.1,
                ch_selection=None,
                feature_kwargs=None,
                use_drop_subject_list=True,
                filter_params=None,
                do_artefact_rejection=True,
                balance_data=True,
                subject_handle=SubjectHandle.INDEPENDENT_DAYS,
                base_dir='.', fast_load=True,
                subjects='all', augment_data=False,
                mode='auto', n_jobs=-3):
    if feature_kwargs is None:
        feature_kwargs = {}

    # removing data which does not affect fast_load
    feature_params = locals().copy()
    feature_params.pop('db_filename')
    feature_params.pop('fast_load')
    feature_params.pop('base_dir')
    feature_params.pop('n_jobs')
    feature_params.pop('mode')

    loader = DataLoader(use_drop_subject_list=use_drop_subject_list,
                        subject_handle=subject_handle,
                        base_config_path=base_dir)
    loader.use_db(db_name)

    if subjects == 'all':
        subject_list = loader.get_subject_list()
    elif isinstance(subjects, list):
        subject_list = [s for s in subjects if not loader.is_subject_in_drop_list(s)]
        assert len(subject_list) > 0, f'All subjects {subjects} are in drop subject list'
    elif isinstance(subjects, int):
        subject_list = loader.get_subject_list()[:subjects]
        feature_params['subjects'] = subject_list
    else:
        raise TypeError(f'subject either must be an integer, list or `all` string')

    database = HDF5Dataset(db_filename, feature_params)

    if not (fast_load and database.exists()):

        tic = time()
        if cpu_count() < CPU_THRESHOLD or mode == 'sequential':  # parallel db generation is slow if there is not enough cpu_cores
            for subj in subject_list:
                files = loader.get_filenames_for_subject(subj)
                artifact_filter = ArtefactFilter(apply_frequency_filter=False) if do_artefact_rejection else None
                windowed_data, labels, subj_ind, ep_ind, orig_mask, fs, info = generate_subject_data(
                    files, loader, subj, filter_params,
                    epoch_tmin, epoch_tmax, window_length, window_step,
                    artifact_filter, balance_data,
                    binarize_labels=db_name is Databases.GAME_PAR_D,
                    augment_data=augment_data,
                    ch_selection=ch_selection
                )
                windowed_data = generate_features(windowed_data, fs, feature_type, info=info, **feature_kwargs)
                database.add_data(windowed_data, labels, subj_ind, ep_ind, orig_mask, fs)
        else:
            subj_db_files = Parallel(n_jobs)(
                delayed(_save_one_subject_data)(feature_type, feature_kwargs, loader, subj,
                                                epoch_tmin, epoch_tmax, window_length, window_step,
                                                filter_params, balance_data,
                                                db_name is Databases.GAME_PAR_D,
                                                do_artefact_rejection,
                                                db_filename.parent,
                                                augment_data,
                                                ch_selection) for subj in subject_list)
            # subj_db_files = list(database.filename.parent.rglob('*_db.hdf5'))
            _merge_database(database, subj_db_files)

        database.close()
        print(f'DB generated under {(time() - tic) / 60:.2f} minutes')

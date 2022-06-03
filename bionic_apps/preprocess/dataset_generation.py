import mne
import numpy as np

from .io import DataLoader, SubjectHandle, get_epochs_from_raw
from ..artifact_filtering.faster import ArtefactFilter
from ..databases import EEG_Databases
from ..feature_extraction import FeatureType, generate_features
from ..handlers.hdf5 import HDF5Dataset
from ..utils import standardize_eeg_channel_names, filter_mne_obj, balance_epoch_nums, _create_binary_label, \
    window_epochs


def generate_eeg_db(db_name, db_filename, f_type=FeatureType.HUGINES,
                    epoch_tmin=0, epoch_tmax=4,
                    window_length=2, window_step=.1,
                    use_drop_subject_list=True,
                    filter_params=None,
                    do_artefact_rejection=True,
                    balance_data=True,
                    subject_handle=SubjectHandle.INDEPENDENT_DAYS):
    if filter_params is None:
        filter_params = {}

    feature_params = locals()
    feature_params['f_type'] = f_type.name
    feature_params['subject_handle'] = subject_handle.name

    loader = DataLoader(use_drop_subject_list=use_drop_subject_list,
                        subject_handle=subject_handle)
    loader.use_db(db_name)

    database = HDF5Dataset(db_filename, feature_params)

    # TODO: handle error
    # Fast load
    if not database.exists():
        for subj in loader.get_subject_list():
            files = loader.get_filenames_for_subject(subj)
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

            if do_artefact_rejection:
                epochs = ArtefactFilter(apply_frequency_filter=False).offline_filter(epochs)

            ep_labels = [list(epochs[i].event_id)[0] for i in range(len(epochs))]

            if balance_data:
                epochs, ep_labels = balance_epoch_nums(epochs, ep_labels)

            if db_name is EEG_Databases.GAME_PAR_D:
                ep_labels = [_create_binary_label(label) for label in ep_labels]

            # window the epochs
            windowed_data = window_epochs(epochs.get_data(),
                                          window_length=window_length, window_step=window_step,
                                          fs=fs)
            del epochs

            groups = [i // windowed_data.shape[1] for i in range(windowed_data.shape[0] * windowed_data.shape[1])]
            labels = [ep_labels[i // windowed_data.shape[1]] for i in range(len(groups))]
            windowed_data = np.vstack(windowed_data)

            windowed_data = generate_features(windowed_data, f_type)
            database.add_data(windowed_data, labels, [subj] * len(labels), groups)
        database.close()

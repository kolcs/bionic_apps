import time
from multiprocessing import Pipe
from pathlib import Path
from warnings import warn, simplefilter

import keras.models
import numpy as np
from sklearn.preprocessing import LabelEncoder

from bionic_apps.ai import ClassifierType, init_classifier
from bionic_apps.artifact_filtering.faster import ArtefactFilter
from bionic_apps.databases import get_eeg_db_name_by_filename, Databases
from bionic_apps.external_connections.lsl.BCI import DSP
from bionic_apps.feature_extraction import FeatureType, get_feature_extractor, generate_features
from bionic_apps.games.braindriver.control import GameControl
from bionic_apps.games.braindriver.logger import GameLogger
from bionic_apps.games.braindriver.opponents import create_opponents
from bionic_apps.handlers.gui import select_files_in_explorer
from bionic_apps.handlers.hdf5 import HDF5Dataset
from bionic_apps.offline_analyses import train_test_subject_data
from bionic_apps.preprocess.dataset_generation import generate_subject_data
from bionic_apps.preprocess.io import DataLoader
from bionic_apps.utils import init_base_config, load_pickle_data, is_platform, save_pickle_data, mask_to_ind
from bionic_apps.validations import validate_feature_classifier_pair

CMD_IN = 1.5  # sec
SUBJ = 0
DB_FILENAME = 'tmp/brain_driver_db.hdf5'
AR_FILTER = 'ar_filter.pkl'

FILTER_PARAMS = dict(
    order=5, l_freq=1, h_freq=45
)

F_TYPE = FeatureType.FFT_RANGE
F_KWARGS = dict(
    fft_low=2, fft_high=40
)

CLF_TYPE = ClassifierType.VOTING_SVM
CLF_KWARGS = dict(
    # C=309.27089776753826, gamma=0.3020223611011116
    # epochs=150, batch_size=32, validation_split=.2, patience=7
)


def start_brain_driver_control_system(feature_type, classifier_type,
                                      epoch_tmin=0, epoch_tmax=4,
                                      window_length=2, window_step=.1, *,
                                      feature_kwargs=None, filter_params=None,
                                      do_artefact_rejection=True, balance_data=True,
                                      augment_data=False,
                                      classifier_kwargs=None,
                                      use_best_clf=True, make_opponents=True,
                                      use_game_logger=True,
                                      eeg_files=None,
                                      db_filename='tmp/brain_driver_db.hdf5',
                                      time_out=None):
    if classifier_kwargs is None:
        classifier_kwargs = {}
    if filter_params is None:
        filter_params = {}
    if feature_kwargs is None:
        feature_kwargs = {}

    db_filename = Path(db_filename)

    feature_type, classifier_type = validate_feature_classifier_pair(feature_type, classifier_type)
    print('Starting BCI System for BrainDriver game...')

    if eeg_files is None:
        eeg_files = select_files_in_explorer(init_base_config(),
                                             message='Select EEG file for training the BCI System!')
        assert len(eeg_files) == 1, 'Please select only one EEG file!'
    elif isinstance(eeg_files, str):
        eeg_files = [eeg_files]

    db_name = get_eeg_db_name_by_filename(eeg_files[0])
    make_binary_classification = db_name is Databases.GAME_PAR_D

    loader = DataLoader()
    loader.use_db(db_name)

    hdf5_f_params = dict(
        feature_type=feature_type,
        feature_kwargs=feature_kwargs,
        filter_params=filter_params,
        epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
        window_length=window_length, window_step=window_step,
        balance_data=balance_data,
        binarize_labels=make_binary_classification,
        do_artefact_rejection=do_artefact_rejection
    )
    for i, file in enumerate(eeg_files):
        hdf5_f_params[f'eegfile{i}'] = file

    base_dir = db_filename.parent
    base_dir.mkdir(parents=True, exist_ok=True)
    database = HDF5Dataset(db_filename, hdf5_f_params)

    if not database.exists():
        artifact_filter = ArtefactFilter(apply_frequency_filter=False) if do_artefact_rejection else None
        windowed_data, labels, subj_ind, ep_ind, orig_mask, fs, info = generate_subject_data(
            eeg_files, loader, SUBJ, filter_params,
            epoch_tmin, epoch_tmax, window_length, window_step,
            artifact_filter, balance_data,
            binarize_labels=make_binary_classification,
            augment_data=augment_data
        )
        windowed_data = generate_features(windowed_data, fs, feature_type, info=info, **feature_kwargs)
        database.add_data(windowed_data, labels, subj_ind, ep_ind, orig_mask, fs)
        database.close()
        save_pickle_data(base_dir.joinpath(AR_FILTER), artifact_filter)
    else:
        artifact_filter = load_pickle_data(base_dir.joinpath(AR_FILTER))

    all_subj = database.get_subject_group()
    y_all = database.get_y()
    le = LabelEncoder()
    y_all = le.fit_transform(y_all)
    fs = database.get_fs()
    subj_ind = mask_to_ind(all_subj == SUBJ)

    cross_acc, clf_filenames = train_test_subject_data(database, subj_ind, classifier_type,
                                                       n_splits=5, shuffle=True,
                                                       save_classifiers=True, label_encoder=le,
                                                       **classifier_kwargs)
    database.close()

    one_hot_output = False
    if use_best_clf:
        best = np.argmax(cross_acc)
        file = clf_filenames[best]
        if Path(file).suffix == '.h5':
            classifier = keras.models.load_model(file)
            one_hot_output = True
        else:
            classifier = load_pickle_data(file)
    else:
        try:
            epochs = classifier_kwargs.pop('epochs')
        except KeyError:
            epochs = None
        try:
            batch_size = classifier_kwargs.pop('batch_size')
        except KeyError:
            batch_size = None

        x = database.get_data(subj_ind)
        y = y_all[subj_ind]
        classifier = init_classifier(classifier_type, x[0].shape, len(le.classes_),
                                     fs=database.get_fs(), **classifier_kwargs)
        if epochs is None:
            classifier.fit(x, y)
        else:
            classifier.fit(x, y, epochs=epochs, batch_size=batch_size)

    database.close()
    parent_conn, child_conn = Pipe()
    if use_game_logger and is_platform('windows'):
        GameLogger(annotator='bv_rcc', data_loader=loader, connection=child_conn).start()

    if make_opponents:
        create_opponents(main_player=1, game_log_conn=parent_conn, reaction=CMD_IN)

    dsp = DSP(use_filter=len(filter_params) > 0, **filter_params)
    assert dsp.fs == fs, 'Sampling rate frequency must be equal for preprocessed and lsl data.'

    controller = GameControl(make_log=True, log_to_stream=True)
    command_converter = loader.get_command_converter() if not make_binary_classification else None

    feature_extractor = get_feature_extractor(feature_type, fs, **feature_kwargs)

    print("Starting game braindriver...")
    simplefilter('always', UserWarning)
    start_time = time.time()
    timestamp = None
    tic = start_time
    while time_out is None or (time.time() - start_time < time_out or timestamp is None):
        timestamp, eeg = dsp.get_eeg_window_in_chunk(window_length)
        if timestamp is not None:
            eeg = np.delete(eeg, -1, axis=0)  # removing last unwanted channel
            eeg = np.array([eeg])

            if do_artefact_rejection:
                eeg = artifact_filter.online_filter(eeg)

            data = feature_extractor.transform(eeg)
            y_pred = classifier.predict(data)
            if one_hot_output:
                y_pred = np.argmax(y_pred, axis=-1)
            y_pred = le.inverse_transform(y_pred)[0]

            if make_binary_classification:
                controller.control_game_with_2_opt(y_pred)
            else:
                command = command_converter[y_pred]
                controller.control_game(command)

            toc = time.time() - tic
            if toc < CMD_IN:
                time.sleep(CMD_IN - toc)
            else:
                warn('Classification took longer than command giving limit!')
            tic = time.time()


def main():
    start_brain_driver_control_system(F_TYPE, CLF_TYPE,
                                      epoch_tmin=0, epoch_tmax=4,
                                      window_length=2, window_step=.1,
                                      feature_kwargs=F_KWARGS, filter_params=FILTER_PARAMS,
                                      do_artefact_rejection=True, balance_data=True,
                                      classifier_kwargs=CLF_KWARGS, use_best_clf=True,
                                      make_opponents=True, use_game_logger=True,
                                      db_filename=DB_FILENAME
                                      )


if __name__ == '__main__':
    main()

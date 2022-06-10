import time
from warnings import warn, simplefilter

import numpy as np

from bionic_apps.ai import ClassifierType, init_classifier
from bionic_apps.artifact_filtering.faster import ArtefactFilter
from bionic_apps.config import SAVE_PATH
from bionic_apps.databases import get_eeg_db_name_by_filename, EEG_Databases
from bionic_apps.external_connections.lsl.BCI import DSP
from bionic_apps.feature_extraction import FeatureType, get_feature_extractor
from bionic_apps.games.braindriver.control import GameControl
from bionic_apps.games.braindriver.logger import GameLogger
from bionic_apps.games.braindriver.opponents import create_opponents
from bionic_apps.handlers.gui import select_files_in_explorer
from bionic_apps.handlers.hdf5 import HDF5Dataset, init_hdf5_db
from bionic_apps.offline_analyses import train_test_data
from bionic_apps.preprocess.dataset_generation import generate_subject_data
from bionic_apps.preprocess.io import DataLoader
from bionic_apps.utils import init_base_config, load_pickle_data, is_platform, save_pickle_data
from bionic_apps.validations import validate_feature_classifier_pair

CMD_IN = 1.5  # sec
DB_FILENAME = 'brain_driver_db.hdf5'
SUBJ = 0

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
)
AR_FILTER = 'ar_filter.pkl'


def start_brain_driver_control_system(epoch_tmin=0, epoch_tmax=4, window_length=2, window_step=.1,
                                      do_artefact_rejection=True, balance_data=True,
                                      use_best_clf=True, make_opponents=True, use_game_logger=True,
                                      eeg_files=None, time_out=None):
    feature_type, classifier_type = validate_feature_classifier_pair(F_TYPE, CLF_TYPE)
    print('Starting BCI System for BrainDriver game...')

    if eeg_files is None:
        eeg_files = select_files_in_explorer(init_base_config(),
                                             message='Select EEG file for training the BCI System!')
        assert len(eeg_files) == 1, 'Please select only one EEG file!'
    elif isinstance(eeg_files, str):
        eeg_files = [eeg_files]

    db_name = get_eeg_db_name_by_filename(eeg_files[0])
    make_binary_classification = db_name is EEG_Databases.GAME_PAR_D

    feature_params = dict(
        feature_type=feature_type,
        feature_kwargs=F_KWARGS,
        filter_params=FILTER_PARAMS,
        epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
        window_length=window_length, window_step=window_step,
        balance_data=balance_data,
        binarize_labels=make_binary_classification
    )

    loader = DataLoader()
    loader.use_db(db_name)

    hdf5_f_params = feature_params.copy()
    hdf5_f_params['do_artefact_rejection'] = do_artefact_rejection
    for i, file in enumerate(eeg_files):
        hdf5_f_params[f'eegfile{i}'] = file

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    database = HDF5Dataset(SAVE_PATH.joinpath(DB_FILENAME), hdf5_f_params)

    if not database.exists():
        artifact_filter = ArtefactFilter(apply_frequency_filter=False) if do_artefact_rejection else None
        subj_data = generate_subject_data(
            eeg_files, loader, SUBJ, artifact_filter=artifact_filter, **feature_params
        )
        database.add_data(*subj_data)
        database.close()
        save_pickle_data(SAVE_PATH.joinpath(AR_FILTER), artifact_filter)
    else:
        artifact_filter = load_pickle_data(SAVE_PATH.joinpath(AR_FILTER))

    db, y_all, subj_ind, ep_ind, le, fs = init_hdf5_db(SAVE_PATH.joinpath(DB_FILENAME))

    x = db.get_data(subj_ind == SUBJ)
    y = y_all[subj_ind == SUBJ]
    groups = ep_ind[subj_ind == SUBJ]

    cross_acc, clf_filenames = train_test_data(classifier_type, x, y, groups=groups, lab_enc=le,
                                               n_splits=5, shuffle=True, save_classifiers=True, **CLF_KWARGS)

    if use_best_clf:
        best = np.argmax(cross_acc)
        file = clf_filenames[best]
        classifier = load_pickle_data(file)
    else:
        try:
            epochs = CLF_KWARGS.pop('epochs')
        except KeyError:
            epochs = None
        classifier = init_classifier(classifier_type, x[0].shape, len(le.classes_), **CLF_KWARGS)
        if epochs is None:
            classifier.fit(x, y)
        else:
            classifier.fit(x, y, epochs=epochs)

    game_log = None
    if use_game_logger and is_platform('windows'):
        from bionic_apps.external_connections.brainvision import RemoteControlClient
        rcc = RemoteControlClient(print_received_messages=False)
        rcc.open_recorder()
        rcc.check_impedance()
        game_log = GameLogger(bv_rcc=rcc)
        game_log.start()

    if make_opponents:
        create_opponents(main_player=1, game_logger=game_log, reaction=CMD_IN)

    dsp = DSP(use_filter=len(FILTER_PARAMS) > 0, **FILTER_PARAMS)
    assert dsp.fs == fs, 'Sampling rate frequency must be equal for preprocessed and lsl data.'

    controller = GameControl(make_log=True, log_to_stream=True, game_logger=game_log)
    command_converter = loader.get_command_converter() if not make_binary_classification else None

    feature_extractor = get_feature_extractor(feature_type, fs, **F_KWARGS)

    print("Starting game braindriver...")
    simplefilter('always', UserWarning)
    start_time = time.time()
    timestamp = None
    tic = start_time
    while time_out is None or (time.time() - start_time < time_out or timestamp is None):
        timestamp, eeg = dsp.get_eeg_window_in_chunk(window_length)
        if timestamp is not None:
            eeg = np.delete(eeg, -1, axis=0)  # removing last unwanted channel

            if do_artefact_rejection:
                eeg = artifact_filter.online_filter(eeg)

            data = feature_extractor.transform(eeg)
            y_pred = classifier.predict(data)
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


if __name__ == '__main__':
    start_brain_driver_control_system()

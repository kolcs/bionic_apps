from ...databases import get_eeg_db_name_by_filename, EEG_Databases
from ...feature_extraction import FeatureType
from ...handlers.gui import select_files_in_explorer
from ...handlers.hdf5 import HDF5Dataset, init_hdf5_db
from ...preprocess.dataset_generation import generate_subject_data
from ...preprocess.io import DataLoader
from ...utils import init_base_config
from ...offline_analyses import train_test_data

CMD_IN = 1.5  # sec
DB_FILENAME = 'brain_driver_db.hdf5'
SUBJ = 0

# FFT_POWERS = [(22, 30), (24, 40), (24, 34)]
#
# # svm params:
# C, GAMMA = 309.27089776753826, 0.3020223611011116

if __name__ == '__main__':
    print('Starting BCI System for game play...')

    eeg_files = select_files_in_explorer(init_base_config(),
                                         message='Select EEG file for training the BCI System!')
    assert len(eeg_files) == 1, 'Please select only one EEG file!'
    eeg_files = eeg_files[0]
    db_name = get_eeg_db_name_by_filename(eeg_files)

    filter_params = dict(
        order=5, l_freq=1, h_freq=45
    )

    # classifier_kwargs = dict(C=C, gamma=GAMMA)

    loader = DataLoader()
    loader.use_db(db_name)

    feature_params = dict(
        feature_type=FeatureType.FFT_RANGE,
        feature_kwargs=dict(fft_low=2, fft_high=40),
        filter_params=filter_params,
        epoch_tmin=0, epoch_tmax=4,
        window_length=2, window_step=.1,
        do_artefact_rejection=True, balance_data=True,
        binarize_labels=db_name is EEG_Databases.GAME_PAR_D
    )
    database = HDF5Dataset(DB_FILENAME, feature_params)

    if not database.exists():
        subj_data = generate_subject_data(
            eeg_files, loader, SUBJ, **feature_params
        )
        database.add_data(*subj_data)
    database.close()

    db, y_all, subj_ind, ep_ind, le = init_hdf5_db(DB_FILENAME)

    x = db.get_data(subj_ind == SUBJ)
    y = y_all[subj_ind == SUBJ]
    groups = ep_ind[subj_ind == SUBJ]

    # todo: cross-train / select best
    cross_acc = train_test_data(classifier_type, x, y, groups=groups, lab_enc=le,
                                n_splits=n_splits, shuffle=True, **classifier_kwargs)

    bci = BCISystem()
    bci.play_game(
        db_name=db_name,
        feature_params=feature_extraction,
        window_length=1,
        pretrain_window_step=0.1,
        epoch_tmax=4,
        command_delay=CMD_IN,
        use_binary_game_logger=True,
        train_file=eeg_files,
        classifier_kwargs=classifier_kwargs,
        make_opponents=True
    )

from BCISystem import BCISystem, FeatureType, Databases
from paramsel import parallel_search_for_fft_params
from preprocess import init_base_config
from gui_handler import select_file_in_explorer

CMD_IN = 1.6  # sec

FFT_POWERS = [(22, 30), (24, 40), (24, 34)]

# svm params:
C, GAMMA = 309.27089776753826, 0.3020223611011116

if __name__ == '__main__':
    print('Starting BCI System for game play...')

    # todo: define it from file selection....
    inp = input('Select paradigm for pilots (C -- 4 way, D -- binary):    C / D\n')
    if inp.upper() == 'C':
        db_name = Databases.GAME_PAR_C
    elif inp.upper() == 'D':
        db_name = Databases.GAME_PAR_D
    else:
        raise NotImplementedError('Can not run BCI System with paradigm {}'.format(inp))

    eeg_file = select_file_in_explorer(init_base_config(),
                                       message='Select EEG file for training the BCI System!')

    classifier_kwargs = dict(C=C, gamma=GAMMA)

    # fft_ranges = FFT_POWERS
    fft_ranges = parallel_search_for_fft_params(
        eeg_file, db_name,
        fft_search_min=14,
        fft_search_max=40,
        fft_search_step=2,
        best_n_fft=7,
        classifier_kwargs=classifier_kwargs
    )

    feature_extraction = dict(
        feature_type=FeatureType.MULTI_FFT_POWER,
        fft_low=14, fft_high=30, fft_step=2, fft_width=2,
        fft_ranges=fft_ranges
    )

    bci = BCISystem()
    bci.play_game(
        db_name=db_name,
        feature_params=feature_extraction,
        window_length=1,
        window_step=0.1,
        epoch_tmax=4,
        command_frequency=CMD_IN,
        use_binary_game_logger=True,
        train_file=eeg_file,
        classifier_kwargs=classifier_kwargs
    )

import time
from enum import Enum
from sys import platform
from warnings import warn, simplefilter

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import ai
import online
from control import GameControl, create_opponents
from logger import setup_logger, log_info, GameLogger
from preprocess import OfflineDataPreprocessor, SubjectKFold, save_pickle_data, init_base_config, Features, \
    make_feature_extraction

AI_MODEL = 'ai.model'
LOGGER_NAME = 'BCISystem'


# db selection options
class Databases(Enum):
    PHYSIONET = 'physionet'
    PILOT_PAR_A = 'pilot_par_a'
    PILOT_PAR_B = 'pilot_par_b'
    TTK = 'ttk'
    GAME = 'game'
    GAME_PAR_C = 'game_par_c'
    GAME_PAR_D = 'game_par_d'


# method selection options
class XvalidateMethod(Enum):
    CROSS_SUBJECT = 'crossSubjectXvalidate'
    SUBJECT = 'subjectXvalidate'
    CROSS_SUBJECT_AND_SAVE_SVM = 'crossSubXandSaveSVM'


# LOG, pandas columns
LOG_COLS = ['Database', 'Method', 'Feature', 'Subject', 'Epoch tmin', 'Epoch tmax', 'Window length', 'Window step',
            'FFT low', 'FFT high', 'FFT step', 'svm_C', 'svm_gamma', 'Accuracy list', 'Avg. Acc']

SUM_NAME = 'weight_sum'


def _generate_table(eeg, filter_list, acc_from=.6, acc_to=1, acc_diff=.01):
    d = pd.DataFrame(eeg[filter_list].groupby(filter_list).count())  # data permutation

    new_cols = list()
    for flow, fhigh in [(acc, acc + acc_diff) for acc in np.arange(acc_from, acc_to, acc_diff)[::-1]]:
        new_cols.append(np.round(flow, 3))
        d[np.round(flow, 3)] = \
            eeg[(eeg['Avg. Acc'] >= flow) & (eeg['Avg. Acc'] < fhigh)].groupby(filter_list, sort=True).count()[
                'Avg. Acc']

    d = d.fillna(0)
    d[SUM_NAME] = np.sum(
        np.array([d[col].array * (col - min(new_cols)) / (max(new_cols) - min(new_cols)) for col in new_cols]),
        axis=0)

    new_cols.insert(0, SUM_NAME)
    d = d.sort_values(new_cols, ascending=[False] * len(new_cols))
    return d


class BCISystem(object):
    """Main class for Brain-Computer Interface application

    This is the main class for the BCI application. Online and offline data manipulation is
    also available.
    """

    def __init__(self, feature=Features.FFT_POWER, make_logs=False):
        """ Constructor for BCI system

        Parameters
        ----------
        feature : Features
            The feature which will be created.
        make_logs : bool
            To make log files or not.
        """
        self._base_dir = init_base_config()
        # self._window_length = window_length
        # self._window_step = window_step
        self._proc = None
        self._prev_timestamp = [0]
        self._ai_model = dict()
        self._log = False
        self._feature = feature
        self._svm_kwargs = dict()

        self._df = None
        self._df_base_data = list()

        if make_logs:
            self._init_log()

    def _init_log(self):
        setup_logger(LOGGER_NAME)
        self._df = pd.DataFrame(columns=LOG_COLS)
        self._log = True

    def _log_and_print(self, msg):
        if self._log:
            log_info(LOGGER_NAME, msg)
        print(msg)

    def _save_params(self, args):
        """Save offline search parameters to pandas table.

        Implemented for '_subject_crossvalidate()' method.

        Parameters
        ----------
        args : list
            The list of parameters to be saved. Should be matched with 'LOG_COLS'
            global parameter.
        """
        if self._log:
            data = self._df_base_data.copy()
            data.extend(args)
            s = pd.Series(data, index=LOG_COLS)
            self._df = self._df.append(s, ignore_index=True)

    def show_results(self, out_file_name='out.csv'):
        """Display and save the parameters of an offline search process.

        The results are saved in a csv file with pandas package.

        Parameters
        ----------
        out_file_name : str
            File name with absolute path. Always should contain ''.csv'' in the end.
        """
        if self._log:
            print(self._df)
            self._df.to_csv(out_file_name, sep=';', encoding='utf-8', index=False)

    def _subject_corssvalidate(self, subject=None, k_fold_num=10):
        """Method for cross-validate classifier results of one subject.

        In each iteration a new classifier is created and new segment of data is given
        to check the consistency of the classification.

        Parameters
        ----------
        subject : int, optional
            Subject number in a given database.
        k_fold_num : int
            The number of cross-validation.
        """
        if subject is None:
            subject = 1
        kfold = SubjectKFold(k_fold_num)

        self._log_and_print("####### Classification report for subject{}: #######".format(subject))
        cross_acc = list()

        for train_x, train_y, test_x, test_y in kfold.split_subject_data(self._proc, subject):
            t = time.time()
            print('Training...')

            # train_x = make_feature_extraction(self._feature, train_x, self._proc._fs, self._proc._fft_low,
            #                                   self._proc._fft_high, self._proc._fft_width, self._proc._fft_step,
            #                                   self._proc._channel_list)
            # test_x = make_feature_extraction(self._feature, test_x, self._proc._fs, self._proc._fft_low,
            #                                   self._proc._fft_high, self._proc._fft_width, self._proc._fft_step,
            #                                   self._proc._channel_list)

            svm = ai.MultiSVM(**self._svm_kwargs)
            svm.fit(train_x, train_y)

            t = time.time() - t
            print("Training elapsed {} seconds.".format(int(t)))

            y_pred = svm.predict(test_x)

            # https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-and-f-measures
            class_report = classification_report(test_y, y_pred)
            conf_martix = confusion_matrix(test_y, y_pred)
            acc = accuracy_score(test_y, y_pred)
            cross_acc.append(acc)

            self._log_and_print("classifier %s:\n%s" % (self, class_report))
            self._log_and_print("Confusion matrix:\n%s\n" % conf_martix)
            self._log_and_print("Accuracy score: {}\n".format(acc))

        self._log_and_print("Avg accuracy: {}".format(np.mean(cross_acc)))
        self._log_and_print("Accuracy scores for k-fold crossvalidation: {}\n".format(cross_acc))
        self._save_params((cross_acc, np.mean(cross_acc)))

    def _crosssubject_crossvalidate(self, subj_n_fold_num=None, save_model=False):
        """Method for cross-validate classifier results between many subjects.

        In each iteration a new classifier is created and new segment of data is given
        to check the consistency of the classification. The cross-validation is made like
        one vs. others way.

        Parameters
        ----------
        subj_n_fold_num : int, optional
            The number of cross-validation. If None is given the cross-validation
            will be made between all subjects in the database.
        """
        kfold = SubjectKFold(subj_n_fold_num)

        for train_x, train_y, test_x, test_y, test_subject in kfold.split(self._proc):
            t = time.time()
            print('Training...')

            svm = ai.MultiSVM(**self._svm_kwargs)
            svm.fit(train_x, train_y)
            t = time.time() - t
            print("Training elapsed {} seconds.".format(int(t)))

            y_pred = svm.predict(test_x)

            class_report = classification_report(test_y, y_pred)
            conf_martix = confusion_matrix(test_y, y_pred)
            acc = accuracy_score(test_y, y_pred)

            # https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-and-f-measures
            self._log_and_print("Classification report for subject{}:".format(test_subject))
            self._log_and_print("classifier %s:\n%s\n" % (self, class_report))
            self._log_and_print("Confusion matrix:\n%s\n" % conf_martix)
            self._log_and_print("Accuracy score: {}\n".format(acc))

            if save_model:
                print("Saving AI model...")
                self._ai_model[test_subject] = svm
                save_pickle_data(self._ai_model, self._proc.proc_db_path + AI_MODEL)
                print("Done\n")

    def _init_db_processor(self, db_name, epoch_tmin=0, epoch_tmax=4, window_length=1, window_step=.25,
                           use_drop_subject_list=True, fast_load=True, make_binary_classification=False,
                           subject=None, select_eeg_file=False, game_file=None):
        """Database initializer.

        Initialize the database preprocessor for the required db, which handles the
        configuration.

        Parameters
        ----------
        db_name : Databases
            Database to work on.
        epoch_tmin : float
            Defining epoch start from trigger signal in seconds.
        epoch_tmax : float
            Defining epoch end from trigger signal in seconds.
        window_length : float
            Length of sliding window in the epochs in seconds.
        window_step : float
            Step of sliding window in seconds.
        use_drop_subject_list : bool
            Whether to use drop subject list from config file or not?
        fast_load : bool
            Handle with extreme care! It loads the result of a previous preprocess task.
        make_binary_classification : bool
            If true the labeling will be converted to binary labels.
        subject : int, list of int, optional
            Data preprocess is made on these subjects.
        select_eeg_file : bool
            Make it True if this function is called during live game.
        game_file : str, optional
            Absolute file path used for parameter selection. This will be only used
            if 'select_eeg_file' is True
        """
        if self._proc is None:

            self._proc = OfflineDataPreprocessor(self._base_dir, epoch_tmin, epoch_tmax, use_drop_subject_list,
                                                 window_length, window_step, fast_load,
                                                 make_binary_classification, subject, select_eeg_file, game_file)
            if db_name == Databases.PHYSIONET:
                self._proc.use_physionet()
            elif db_name == Databases.PILOT_PAR_A:
                self._proc.use_pilot_par_a()
            elif db_name == Databases.PILOT_PAR_B:
                self._proc.use_pilot_par_b()
            elif db_name == Databases.TTK:
                self._proc.use_ttk_db()
            elif db_name == Databases.GAME:
                self._proc.use_game_data()
            elif db_name == Databases.GAME_PAR_C:
                self._proc.use_game_par_c()
            elif db_name == Databases.GAME_PAR_D:
                self._proc.use_game_par_d()

            else:
                raise NotImplementedError('Database processor for {} db is not implemented'.format(db_name))

    def clear_db_processor(self):
        """Removes data preprocessor with all preprocessed data"""
        self._proc = None

    def offline_processing(self, db_name=Databases.PHYSIONET, feature=None, fft_low=7, fft_high=13, fft_step=2,
                           fft_width=2, method=XvalidateMethod.SUBJECT, epoch_tmin=0, epoch_tmax=3, window_length=0.5,
                           window_step=0.25, subject=None, use_drop_subject_list=True, fast_load=False,
                           subj_n_fold_num=None, make_binary_classification=False, channel_list=None, reuse_data=False,
                           **svm_kwargs):
        """Offline data processing.

        This method creates an offline BCI-System which make the data preprocessing
        and calculates the classification results.

        Parameters
        ----------
        db_name : Databases
            The database which will be used.
        feature : Features, optional
            Specify the features which will be created in the preprocessing phase.
        fft_low : float or list of (float, float)
            FFT parameters for frequency features. If list of tuples of 2 floats is given
            it is interpreted as a list of specified frequency ranges with low and high
            boundaries.
        fft_high, fft_step, fft_width : float
            FFT parameters for frequency features. 
        method : XvalidateMethod
            The type of cross-validation
        epoch_tmin : float
            Defining epoch start from trigger signal in seconds.
        epoch_tmax : float
            Defining epoch end from trigger signal in seconds.
        window_length : float
            Length of sliding window in the epochs in seconds.
        window_step : float
            Step of sliding window in seconds.
        subject : int, optional
            Subject number in a given database.
        use_drop_subject_list : bool
            Whether to use drop subject list from config file or not?
        fast_load : bool
            Handle with extreme care! It loads the result of a previous preprocess task.
        subj_n_fold_num : int, optional
            The number of cross-validation. If None is given the cross-validation
            will be made between all subjects in the database.
        make_binary_classification : bool
            If true the labeling will be converted to binary labels.
        channel_list : list of int
            Dummy eeg channel selection. Do not use it.
        reuse_data : bool
            Preprocess methods will be omitted if True. Use it for classifier 
            hyper-parameter selection only.
        svm_kwargs : dict
             Arbitrary keyword arguments for SVM
        """
        if feature is not None:
            self._feature = feature
        self._svm_kwargs = svm_kwargs

        self._init_db_processor(db_name, epoch_tmin, epoch_tmax, window_length, window_step,
                                use_drop_subject_list, fast_load, make_binary_classification, subject)

        self._proc.run(self._feature, fft_low, fft_high, fft_step, fft_width, channel_list, reuse_data)

        if self._log:
            self._df_base_data = [db_name.name, method, self._feature.name, subject,
                                  epoch_tmin, epoch_tmax,
                                  window_length, window_step,
                                  fft_low, fft_high, fft_step,
                                  svm_kwargs.get('C'), svm_kwargs.get('gamma')
                                  ]

        if method == XvalidateMethod.CROSS_SUBJECT:
            self._crosssubject_crossvalidate(subj_n_fold_num)

        elif method == XvalidateMethod.CROSS_SUBJECT_AND_SAVE_SVM:
            self._crosssubject_crossvalidate(save_model=True)

        elif method == XvalidateMethod.SUBJECT:
            self._subject_corssvalidate(subject, subj_n_fold_num)

        else:
            raise NotImplementedError('Method {} is not implemented'.format(method))

    def _search_for_fft_params(self, db_name,
                               fft_min, fft_max, fft_search_step,
                               epoch_tmin, epoch_tmax,
                               window_length, window_step,
                               make_binary_classification,
                               best_n_fft=7):
        """Pre-search for best FFT Power ranges"""
        from gui_handler import select_file_in_explorer
        train_file = select_file_in_explorer(self._base_dir)
        for fft_low in range(fft_min, fft_max - 2, fft_search_step):
            for fft_high in range(fft_low + 2, fft_max, fft_search_step):
                self._df_base_data = [db_name.name, XvalidateMethod.SUBJECT.value, self._feature.name, 0,
                                      epoch_tmin, epoch_tmax,
                                      window_length, window_step,
                                      fft_low, fft_high, 2,
                                      self._svm_kwargs.get('C'), self._svm_kwargs.get('gamma')
                                      ]
                self._init_db_processor(db_name=db_name, epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
                                        window_length=window_length, window_step=window_step,
                                        use_drop_subject_list=False, fast_load=False,
                                        make_binary_classification=make_binary_classification,
                                        select_eeg_file=True, game_file=train_file)
                self._proc.run(Features.FFT_POWER, fft_low, fft_high)
                train_file = self._proc.eeg_file
                self._subject_corssvalidate(subject=0, k_fold_num=5)
                self.clear_db_processor()

        res = _generate_table(self._df[['FFT low', 'FFT high', 'Avg. Acc']],
                              ['FFT low', 'FFT high'])
        fft_list = [res.index[i] for i in range(min(len(res.index), best_n_fft))]

        return train_file, fft_list

    def play_game(self, db_name=Databases.GAME, feature=None,
                  fft_low=7, fft_high=13,
                  epoch_tmin=0, epoch_tmax=0,
                  window_length=1, window_step=.1,
                  command_frequency=0.5,
                  make_binary_classification=False,
                  use_binary_game_logger=False,
                  make_opponents=False,
                  make_fft_param_selection=False,
                  fft_search_min=14, fft_search_max=40, fft_search_step=4,
                  best_n_fft=7,
                  **svm_kwargs):
        """Function for online BCI game and control.
        
        Parameters
        ----------
        db_name : Databases
            The database which will be used.
        feature : Features, optional
            Specify the features which will be created in the preprocessing phase.
        fft_low : float or list of (float, float)
            FFT parameters for frequency features. If list of tuples of 2 floats is given
            it is interpreted as a list of specified frequency ranges with low and high
            boundaries.
        fft_high: float
            FFT parameters for frequency features. 
        epoch_tmin : float
            Defining epoch start from trigger signal in seconds.
        epoch_tmax : float
            Defining epoch end from trigger signal in seconds.
        window_length : float
            Length of sliding window in the epochs in seconds.
        window_step : float
            Step of sliding window in seconds.
        command_frequency : float 
            The frequency of given commands in second.
        make_binary_classification : bool
            If true the labeling will be converted to binary labels.
        use_binary_game_logger : bool
            If True game events will be sent to BrainVision Amplifier.
        make_opponents : bool
            Artificial opponents for game player.
        make_fft_param_selection : bool
            Pre-search for best FFT Power ranges.
        fft_search_min, fft_search_max, fft_search_step : float
            Parameters for best FFT Power range search.
        best_n_fft : int
            The number of FFT Power ranges which will be selected.
        svm_kwargs : dict
             Arbitrary keyword arguments for SVM
        """

        if feature is not None:
            self._feature = feature
        if db_name == Databases.GAME_PAR_D:
            make_binary_classification = True
        self._svm_kwargs = svm_kwargs

        train_file = None
        if make_fft_param_selection and self._feature == Features.MULTI_FFT_POWER:
            self._init_log()
            t = time.time()
            train_file, fft_low = self._search_for_fft_params(db_name, fft_min=fft_search_min, fft_max=fft_search_max,
                                                              fft_search_step=fft_search_step,
                                                              epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
                                                              window_length=window_length, window_step=window_step,
                                                              make_binary_classification=make_binary_classification,
                                                              best_n_fft=best_n_fft)
            print('Parameter selection took {:.2f} min'.format((time.time() - t) / 60))
        elif make_fft_param_selection:
            raise ValueError(
                'FFT parameter selection only available for MULTI_FFT_POWER, {} were selected instead'.format(
                    self._feature.name))

        # print(train_file, fft_low)
        # exit(12)
        self._init_db_processor(db_name=db_name, epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
                                window_length=window_length, window_step=window_step,
                                use_drop_subject_list=False, fast_load=False,
                                make_binary_classification=make_binary_classification,
                                select_eeg_file=True, game_file=train_file)
        self._proc.run(self._feature, fft_low, fft_high)
        print('Training...')
        t = time.time()
        data, labels = self._proc.get_subject_data(0)
        svm = ai.MultiSVM(**self._svm_kwargs)
        svm.fit(data, labels)
        print("Training elapsed {} seconds.".format(int(time.time() - t)))

        game_log = None
        if use_binary_game_logger and platform.startswith('win'):
            from brainvision import RemoteControlClient
            rcc = RemoteControlClient(print_received_messages=False)
            rcc.open_recorder()
            rcc.check_impedance()
            game_log = GameLogger(bv_rcc=rcc)
            game_log.start()

        if make_opponents:
            create_opponents(main_player=1, game_logger=game_log, reaction=command_frequency)

        dsp = online.DSP()

        controller = GameControl(make_log=True, log_to_stream=True, game_logger=game_log)
        command_converter = self._proc.get_command_converter() if not make_binary_classification else dict()

        print("Starting game control...")
        simplefilter('always', UserWarning)
        while True:
            timestamp, eeg = dsp.get_eeg_window_in_chunk(window_length)
            if timestamp is not None:
                tic = time.time()
                eeg = np.delete(eeg, -1, axis=0)  # removing last unwanted channel

                data = make_feature_extraction(self._feature, eeg, fs=dsp.fs, fft_low=fft_low, fft_high=fft_high)
                y_pred = svm.predict(data)[0]

                if make_binary_classification:
                    controller.control_game_with_2_opt(y_pred)
                else:
                    command = command_converter[y_pred]
                    controller.control_game(command)

                toc = time.time() - tic
                if toc < command_frequency:
                    time.sleep(command_frequency - toc)
                else:
                    warn('Classification took longer than command giving limit!')


if __name__ == '__main__':
    # only for MULTI_FFT_POWER, tested by grid-search...
    fft_powers = [(14, 36), (18, 32), (18, 36), (22, 28),
                  (22, 32), (22, 36), (26, 32), (26, 36)]

    bci = BCISystem()
    bci.offline_processing(db_name=Databases.GAME_PAR_D,
                           feature=Features.MULTI_FFT_POWER,
                           fast_load=False,
                           epoch_tmin=0, epoch_tmax=4,
                           fft_low=fft_powers, fft_high=40, fft_step=2, fft_width=2,
                           window_length=1, window_step=.1,
                           method=XvalidateMethod.SUBJECT,
                           subject=1,
                           use_drop_subject_list=True,
                           subj_n_fold_num=5,
                           make_binary_classification=True)

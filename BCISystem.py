import time
from enum import Enum
from sys import platform
from warnings import warn, simplefilter

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import online
from ai import init_classifier, ClassifierType
from control import GameControl, create_opponents
from logger import setup_logger, log_info, GameLogger
from preprocess import OfflineDataPreprocessor, SubjectKFold, save_pickle_data, init_base_config, FeatureType, \
    make_feature_extraction, Databases

AI_MODEL = 'ai.model'
LOGGER_NAME = 'BCISystem'


# method selection options
class XvalidateMethod(Enum):
    CROSS_SUBJECT = 'crossSubjectXvalidate'
    SUBJECT = 'subjectXvalidate'
    CROSS_SUBJECT_AND_SAVE_SVM = 'crossSubXandSaveSVM'


# LOG, pandas columns
LOG_COLS = ['Database', 'Method', 'Feature', 'Subject', 'Epoch tmin', 'Epoch tmax', 'Window length', 'Window step',
            'FFT low', 'FFT high', 'FFT step', 'FFT ranges', 'svm_C', 'svm_gamma', 'Accuracy list', 'Avg. Acc']


class BCISystem(object):
    """Main class for Brain-Computer Interface application.

    This is the main class for the BCI application. Online and offline data manipulation is
    also available.
    """

    def __init__(self, make_logs=False):
        """Constructor for BCI system.

        Parameters
        ----------
        make_logs : bool
            To make log files or not.
        """
        self._base_dir = init_base_config()
        self._proc = None
        self._ai_model = dict()
        self._log = False

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

    def _subject_corssvalidate(self, subject=None, k_fold_num=10,
                               classifier_type=ClassifierType.SVM, classifier_kwargs=None):
        """Method for cross-validate classifier results of one subject.

        In each iteration a new classifier is created and new segment of data is given
        to check the consistency of the classification.

        Parameters
        ----------
        subject : int, optional
            Subject number in a given database.
        k_fold_num : int
            The number of cross-validation.
        classifier_type : ClassifierType
            The type of the classifier.
        classifier_kwargs : dict, optional
             Arbitrary keyword arguments for classifier.
        """
        if classifier_kwargs is None:
            classifier_kwargs = {}
        if subject is None:
            subject = 1
        kfold = SubjectKFold(k_fold_num)

        self._log_and_print("####### Classification report for subject{}: #######".format(subject))
        cross_acc = list()

        for train_x, train_y, test_x, test_y in kfold.split_subject_data(self._proc, subject):
            t = time.time()
            print('Training...')

            # train_x = make_feature_extraction(self._proc.feature_type, train_x, self._proc.fs,
            #                                   **self._proc.feature_kwargs)
            # test_x = make_feature_extraction(self._proc.feature_type, test_x, self._proc.fs,
            #                                  **self._proc.feature_kwargs)

            classifier = init_classifier(classifier_type, **classifier_kwargs)
            classifier.fit(train_x, train_y)

            t = time.time() - t
            print("Training elapsed {} seconds.".format(int(t)))

            y_pred = classifier.predict(test_x)

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

    def _crosssubject_crossvalidate(self, subj_n_fold_num=None, save_model=False,
                                    classifier_type=ClassifierType.SVM, classifier_kwargs=None):
        """Method for cross-validate classifier results between many subjects.

        In each iteration a new classifier is created and new segment of data is given
        to check the consistency of the classification. The cross-validation is made like
        one vs. others way.

        Parameters
        ----------
        subj_n_fold_num : int, optional
            The number of cross-validation. If None is given the cross-validation
            will be made between all subjects in the database.
        classifier_type : ClassifierType
            The type of the classifier.
        classifier_kwargs : dict, optional
             Arbitrary keyword arguments for classifier.
        """
        if classifier_kwargs is None:
            classifier_kwargs = {}
        kfold = SubjectKFold(subj_n_fold_num)

        for train_x, train_y, test_x, test_y, test_subject in kfold.split(self._proc):
            t = time.time()
            print('Training...')

            classifier = init_classifier(classifier_type, **classifier_kwargs)
            classifier.fit(train_x, train_y)
            t = time.time() - t
            print("Training elapsed {} seconds.".format(int(t)))

            y_pred = classifier.predict(test_x)

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
                self._ai_model[test_subject] = classifier
                save_pickle_data(self._ai_model, str(self._proc.proc_db_path.joinpath(AI_MODEL)))
                print("Done\n")

    def _init_db_processor(self, db_name, epoch_tmin=0, epoch_tmax=4, window_length=1, window_step=.25,
                           use_drop_subject_list=True, fast_load=True, make_binary_classification=False,
                           subject=None, select_eeg_file=False, train_file=None):
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
        train_file : str, optional
            Absolute file path used for parameter selection. This will be only used
            if 'select_eeg_file' is True
        """
        if self._proc is None:
            self._proc = OfflineDataPreprocessor(self._base_dir, epoch_tmin, epoch_tmax, use_drop_subject_list,
                                                 window_length, window_step, fast_load,
                                                 make_binary_classification, subject, select_eeg_file, train_file)
            self._proc.use_db(db_name)

    def clear_db_processor(self):
        """Removes data preprocessor with all preprocessed data"""
        self._proc = None

    def offline_processing(self, db_name=Databases.PHYSIONET, feature_params=None,
                           epoch_tmin=0, epoch_tmax=3,
                           window_length=0.5, window_step=0.25,
                           method=XvalidateMethod.SUBJECT,
                           subject=None, use_drop_subject_list=True, fast_load=False,
                           subj_n_fold_num=None, make_binary_classification=False, reuse_data=False,
                           train_file=None,
                           classifier_type=ClassifierType.SVM, classifier_kwargs=None):
        """Offline data processing.

        This method creates an offline BCI-System which make the data preprocessing
        and calculates the classification results.

        Parameters
        ----------
        db_name : Databases
            The database which will be used.
        feature_params : dict
            Arbitrary keyword arguments for feature extraction.
        epoch_tmin : float
            Defining epoch start from trigger signal in seconds.
        epoch_tmax : float
            Defining epoch end from trigger signal in seconds.
        window_length : float
            Length of sliding window in the epochs in seconds.
        window_step : float
            Step of sliding window in seconds.
        method : XvalidateMethod
            The type of cross-validation
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
        reuse_data : bool
            Preprocess methods will be omitted if True. Use it for classifier
            hyper-parameter selection only.
        train_file : str, optional
            File to train on.
        classifier_type : ClassifierType
            The type of the classifier.
        classifier_kwargs : dict, optional
             Arbitrary keyword arguments for classifier.
        """
        if classifier_kwargs is None:
            classifier_kwargs = {}
        assert feature_params is not None, 'Feature parameters must be defined.'

        if db_name == Databases.GAME_PAR_D:
            make_binary_classification = True

        select_eeg_file = False
        if train_file is str:
            select_eeg_file = True
            subject = 0

        self._init_db_processor(db_name, epoch_tmin, epoch_tmax, window_length, window_step,
                                use_drop_subject_list, fast_load, make_binary_classification, subject,
                                select_eeg_file=select_eeg_file, train_file=train_file)

        self._proc.run(reuse_data=reuse_data, **feature_params)

        if self._log:
            self._df_base_data = [db_name.name, method, feature_params.get('feature_type').name, subject,
                                  epoch_tmin, epoch_tmax,
                                  window_length, window_step,
                                  feature_params.get('fft_low'), feature_params.get('fft_high'),
                                  feature_params.get('fft_step'), feature_params.get('fft_ranges'),
                                  classifier_kwargs.get('C'), classifier_kwargs.get('gamma')
                                  ]

        if method == XvalidateMethod.CROSS_SUBJECT:
            self._crosssubject_crossvalidate(subj_n_fold_num, classifier_type=classifier_type,
                                             classifier_kwargs=classifier_kwargs)

        elif method == XvalidateMethod.CROSS_SUBJECT_AND_SAVE_SVM:
            self._crosssubject_crossvalidate(save_model=True, classifier_type=classifier_type,
                                             classifier_kwargs=classifier_kwargs)

        elif method == XvalidateMethod.SUBJECT:
            self._subject_corssvalidate(subject, subj_n_fold_num,
                                        classifier_type=classifier_type, classifier_kwargs=classifier_kwargs)

        else:
            raise NotImplementedError('Method {} is not implemented'.format(method))

    def play_game(self, db_name=Databases.GAME, feature_params=None,
                  epoch_tmin=0, epoch_tmax=4,
                  window_length=1, window_step=.1,
                  command_frequency=0.5,
                  make_binary_classification=False,
                  use_binary_game_logger=False,
                  make_opponents=False,
                  train_file=None,
                  classifier_type=ClassifierType.SVM,
                  classifier_kwargs=None):
        """Function for online BCI game and control.

        Parameters
        ----------
        db_name : Databases
            The database which will be used.
        feature_params : dict
            Arbitrary keyword arguments for feature extraction.
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
        train_file : str, optional
            File to train on. Use it only for test function.
        classifier_type : ClassifierType
            The type of the classifier.
        classifier_kwargs : dict, optional
             Arbitrary keyword arguments for classifier.
        """

        if classifier_kwargs is None:
            classifier_kwargs = {}
        assert feature_params is not None, 'Feature parameters must be defined.'

        if db_name == Databases.GAME_PAR_D:
            make_binary_classification = True

        self._init_db_processor(db_name=db_name, epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
                                window_length=window_length, window_step=window_step,
                                use_drop_subject_list=False, fast_load=False,
                                make_binary_classification=make_binary_classification,
                                select_eeg_file=True, train_file=train_file)
        self._proc.run(**feature_params)
        print('Training...')
        t = time.time()
        data, labels = self._proc.get_subject_data(0)
        classifier = init_classifier(classifier_type, **classifier_kwargs)
        classifier.fit(data, labels)
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
        tic = time.time()
        while True:
            timestamp, eeg = dsp.get_eeg_window_in_chunk(window_length)
            if timestamp is not None:
                eeg = np.delete(eeg, -1, axis=0)  # removing last unwanted channel

                data = make_feature_extraction(data=eeg, fs=dsp.fs, **feature_params)
                y_pred = classifier.predict(data)[0]

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
                tic = time.time()


if __name__ == '__main__':
    # only for MULTI_FFT_POWER, tested by grid-search...
    fft_powers = [(14, 36), (18, 32), (18, 36), (22, 28),
                  (22, 32), (22, 36), (26, 32), (26, 36)]

    feature_extraction = dict(
        feature_type=FeatureType.MULTI_FFT_POWER,
        fft_low=14, fft_high=30, fft_step=2, fft_width=2, fft_ranges=fft_powers
    )

    bci = BCISystem()
    bci.offline_processing(db_name=Databases.GAME_PAR_D,
                           feature_params=feature_extraction,
                           fast_load=False,
                           epoch_tmin=0, epoch_tmax=4,
                           window_length=1, window_step=.1,
                           method=XvalidateMethod.SUBJECT,
                           subject=1,
                           use_drop_subject_list=True,
                           subj_n_fold_num=5,
                           make_binary_classification=True)

import time
from enum import Enum
from sys import platform
from warnings import warn, simplefilter

import numpy as np
import pandas as pd
from mne import set_log_level as mne_set_log_level
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

import online
from ai import init_classifier, ClassifierType
from control import GameControl, create_opponents
from logger import setup_logger, log_info, GameLogger
from preprocess import OfflineDataPreprocessor, SubjectKFold, init_base_config, FeatureType, \
    FeatureExtractor, Databases, DataHandler

AI_MODEL = 'ai.model'
LOGGER_NAME = 'BCISystem'


# method selection options
class XvalidateMethod(Enum):
    CROSS_SUBJECT = 'crossSubjectXvalidate'
    SUBJECT = 'subjectXvalidate'


# LOG, pandas columns
LOG_COLS = ['Database', 'Method', 'Feature', 'Subject', 'Epoch tmin', 'Epoch tmax', 'Window length', 'Window step',
            'FFT low', 'FFT high', 'FFT step', 'FFT ranges', 'svm_C', 'svm_gamma', 'Accuracy list', 'Avg. Acc']


def _validate_feature_classifier_pair(feature_type, classifier_type):
    if classifier_type is ClassifierType.SVM and feature_type in \
            [FeatureType.FFT_POWER, FeatureType.FFT_RANGE, FeatureType.MULTI_FFT_POWER]:
        valid = True
    elif classifier_type is ClassifierType.CASCADE_CONV_REC and feature_type is FeatureType.SPATIAL_TEMPORAL:
        valid = True
    elif classifier_type in [ClassifierType.DENSE_NET_121, ClassifierType.DENSE_NET_169, ClassifierType.DENSE_NET_201,
                             ClassifierType.VGG16, ClassifierType.VGG19] and \
            feature_type in [FeatureType.MULTI_FFT_POWER, FeatureType.FFT_RANGE, FeatureType.SPATIAL_FFT_POWER]:
        valid = True
    else:
        valid = False
    assert valid, 'Feature {} is not implemented for classifier {}'.format(feature_type.name, classifier_type.name)


class BCISystem(object):
    """Main class for Brain-Computer Interface application.

    This is the main class for the BCI application. Online and offline data manipulation is
    also available.
    """

    def __init__(self, make_logs=False, verbose=True):
        """Constructor for BCI system.

        Parameters
        ----------
        make_logs : bool
            To make log files or not.
        """
        self._base_dir = init_base_config()
        self._proc = OfflineDataPreprocessor('.')
        self._log = False

        self._df = None
        self._df_base_data = list()
        self._verbose = verbose

        if make_logs:
            self._init_log()
        mne_set_log_level(verbose)

    def _init_log(self):
        setup_logger(LOGGER_NAME, verbose=self._verbose)
        self._df = pd.DataFrame(columns=LOG_COLS)
        self._log = True

    def _log_and_print(self, msg):
        if self._log:
            log_info(LOGGER_NAME, msg)
        if self._verbose:
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

    def log_results(self, out_file_name='out.csv'):
        """Display and save the parameters of an offline search process.

        The results are saved in a csv file with pandas package.

        Parameters
        ----------
        out_file_name : str
            File name with absolute path. Always should contain ''.csv'' in the end.
        """
        if self._log:
            if self._verbose:
                print(self._df)
            self._df.to_csv(out_file_name, sep=';', encoding='utf-8', index=False)

    @staticmethod
    def _train_classifier(train, validation, classifier_type, input_shape, output_classes, classifier_kwargs,
                          label_encoder, make_binary_classification, batch_size=None):

        classifier = init_classifier(classifier_type, output_classes, input_shape,
                                     **classifier_kwargs)

        train_ds = DataHandler(train, label_encoder, make_binary_classification).get_tf_dataset()

        if classifier_type == ClassifierType.SVM:
            train_x, train_y = zip(*train_ds.as_numpy_iterator())
            train_x = np.array(train_x)
            train_y = np.array(train_y)
            classifier.fit(train_x, train_y)
        else:
            train_ds = train_ds.batch(batch_size).prefetch()
            if validation is not None:
                val_ds = DataHandler(validation, label_encoder, make_binary_classification).get_tf_dataset()
                classifier.fit(train_ds, validation_data=val_ds)
            else:
                classifier.fit(train_ds)
        return classifier

    def offline_processing(self, db_name=Databases.PHYSIONET, feature_params=None,
                           epoch_tmin=0, epoch_tmax=4,
                           window_length=1.0, window_step=.1,
                           method=XvalidateMethod.SUBJECT,
                           subject=None, use_drop_subject_list=True, fast_load=True,
                           subj_n_fold_num=None, shuffle_data=True,
                           make_binary_classification=False, train_file=None,
                           classifier_type=ClassifierType.SVM, classifier_kwargs=None, batch_size=None):
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
        subject : int or None
            Subject number in a given database.
        use_drop_subject_list : bool
            Whether to use drop subject list from config file or not?
        fast_load : bool
            Handle with extreme care! It loads the result of a previous preprocess task.
        subj_n_fold_num : int, optional
            The number of cross-validation. If None is given the cross-validation
            will be made between all subjects in the database.
        shuffle_data : bool
            Shuffle or not the order of epochs in each given task.
        make_binary_classification : bool
            If true the labeling will be converted to binary labels.
        train_file : str, optional
            File to train on.
        classifier_type : ClassifierType
            The type of the classifier.
        classifier_kwargs : dict, optional
             Arbitrary keyword arguments for classifier.
        batch_size : int or None
            Define if batch size data generation is required. Otherwise all
            features will be created before training.
        """
        if classifier_kwargs is None:
            classifier_kwargs = {}
        assert feature_params is not None, 'Feature parameters must be defined.'
        _validate_feature_classifier_pair(feature_params['feature_type'], classifier_type)

        if db_name == Databases.GAME_PAR_D:
            make_binary_classification = True

        select_eeg_file = False
        if train_file is str:
            select_eeg_file = True
            subject = 0

        if method == XvalidateMethod.CROSS_SUBJECT:
            subject = None
        elif method == XvalidateMethod.SUBJECT:
            if subject is None:
                subject = 1
            print('{}, {} - Subject{}'.format(db_name.name, feature_params.get('feature_type').name, subject))
        else:
            raise NotImplementedError('Method {} is not implemented'.format(method))

        self._proc = OfflineDataPreprocessor(self._base_dir, epoch_tmin, epoch_tmax, window_length, window_step,
                                             use_drop_subject_list=use_drop_subject_list, fast_load=fast_load,
                                             subject=subject, select_eeg_file=select_eeg_file, eeg_file=train_file)
        self._proc.use_db(db_name).run(**feature_params)
        assert len(self._proc.get_subjects()) > 0, 'There are no preprocessed subjects...'

        if self._log:
            self._df_base_data = [db_name.name, method, feature_params.get('feature_type').name, subject,
                                  epoch_tmin, epoch_tmax,
                                  window_length, window_step,
                                  feature_params.get('fft_low'), feature_params.get('fft_high'),
                                  feature_params.get('fft_step'), feature_params.get('fft_ranges'),
                                  classifier_kwargs.get('C'), classifier_kwargs.get('gamma')
                                  ]

        kfold = SubjectKFold(self._proc, subj_n_fold_num, shuffle_data=shuffle_data)

        cross_acc = list()

        labels = self._proc.get_labels(make_binary_classification)
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)

        for train, test, val, subject in kfold.split(subject):
            t = time.time()
            if self._verbose:
                print('Training...')

            classifier = self._train_classifier(train, val, classifier_type, self._proc.get_feature_shape(),
                                                len(labels), classifier_kwargs, label_encoder,
                                                make_binary_classification, batch_size)

            t = time.time() - t
            if self._verbose:
                print("Training elapsed {} seconds.".format(int(t)))

            test_ds = DataHandler(test, label_encoder, make_binary_classification).get_tf_dataset()
            test_x, test_y = zip(*test_ds.as_numpy_iterator())
            test_x = np.array(test_x)
            test_y = np.array(test_y)

            if classifier_type != ClassifierType.SVM:
                test_ds = test_ds.batch(batch_size).prefetch()
                classifier.evaluate(test_ds)  # todo: do we need dataset for testing?

            y_pred = classifier.predict(test_x)
            y_pred = label_encoder.inverse_transform(y_pred)
            test_y = label_encoder.inverse_transform(test_y)

            # https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-and-f-measures
            class_report = classification_report(test_y, y_pred)
            conf_martix = confusion_matrix(test_y, y_pred)
            acc = accuracy_score(test_y, y_pred)
            cross_acc.append(acc)

            self._log_and_print("####### Classification report for subject{}: #######".format(subject))
            self._log_and_print("classifier %s:\n%s" % (self, class_report))
            self._log_and_print("Confusion matrix:\n%s\n" % conf_martix)
            self._log_and_print("Accuracy score: {}\n".format(acc))

        self._log_and_print("Avg accuracy: {}".format(np.mean(cross_acc)))
        self._log_and_print("Accuracy scores for k-fold crossvalidation: {}\n".format(cross_acc))
        self._save_params((cross_acc, np.mean(cross_acc)))

    def play_game(self, db_name=Databases.GAME, feature_params=None,
                  epoch_tmin=0, epoch_tmax=4,
                  window_length=1,
                  command_delay=0.5,
                  make_binary_classification=False,
                  use_binary_game_logger=False,
                  make_opponents=False,
                  train_file=None,
                  classifier_type=ClassifierType.SVM,
                  classifier_kwargs=None,
                  batch_size=None,
                  time_out=None):
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
        command_delay : float
            Each command can be performed after each t seconds.
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
        batch_size : int or None
            Define if batch size data generation is required. Otherwise all
            features will be created before training.
        time_out : float, optional
            Use it only at testing.
        """

        if classifier_kwargs is None:
            classifier_kwargs = {}
        assert feature_params is not None, 'Feature parameters must be defined.'
        _validate_feature_classifier_pair(feature_params['feature_type'], classifier_type)

        if db_name == Databases.GAME_PAR_D:
            make_binary_classification = True

        self._proc = OfflineDataPreprocessor(self._base_dir, epoch_tmin, epoch_tmax, use_drop_subject_list=False,
                                             fast_load=False, select_eeg_file=True, eeg_file=train_file)
        self._proc.use_db(db_name).run(**feature_params)
        print('Training...')
        t = time.time()

        feature_extractor = FeatureExtractor(fs=self._proc.fs, info=self._proc.info, **feature_params)

        data_list = self._proc.get_processed_db_source(0, only_files=True)
        labels = self._proc.get_labels(make_binary_classification)

        label_encoder = LabelEncoder()
        label_encoder.fit(labels)

        classifier = self._train_classifier(data_list, None, classifier_type, self._proc.get_feature_shape(),
                                            len(labels), classifier_kwargs, label_encoder, make_binary_classification,
                                            batch_size)

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
            create_opponents(main_player=1, game_logger=game_log, reaction=command_delay)

        dsp = online.DSP()
        assert dsp.fs == self._proc.fs, 'Sampling rate frequency must be equal for preprocessed and online data.'

        controller = GameControl(make_log=True, log_to_stream=True, game_logger=game_log)
        command_converter = self._proc.get_command_converter() if not make_binary_classification else dict()

        print("Starting game control...")
        simplefilter('always', UserWarning)
        start_time = time.time()
        tic = start_time
        while time_out is None or time.time() - start_time < time_out:
            timestamp, eeg = dsp.get_eeg_window_in_chunk(window_length)
            if timestamp is not None:
                eeg = np.delete(eeg, -1, axis=0)  # removing last unwanted channel

                data = feature_extractor.run(eeg)
                y_pred = classifier.predict(data)
                y_pred = label_encoder.inverse_transform(y_pred)[0]

                if make_binary_classification:
                    controller.control_game_with_2_opt(y_pred)
                else:
                    command = command_converter[y_pred]
                    controller.control_game(command)

                toc = time.time() - tic
                if toc < command_delay:
                    time.sleep(command_delay - toc)
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

import time
from enum import Enum
from multiprocessing import Queue, Process
from warnings import warn, simplefilter

import numpy as np
import pandas as pd
from mne import set_log_level as mne_set_log_level
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow import data as tf_data
from tensorflow import test as tf_test

import online
from ai import init_classifier, ClassifierType
from control import GameControl, create_opponents
from logger import setup_logger, log_info, GameLogger
from preprocess import OfflineDataPreprocessor, OnlineDataPreprocessor, SubjectKFold, \
    FeatureType, FeatureExtractor, Databases, DataHandler, is_platform, SubjectHandle, DataLoader

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
            [FeatureType.AVG_FFT_POWER, FeatureType.FFT_RANGE, FeatureType.MULTI_AVG_FFT_POW]:
        valid = True
    elif classifier_type is ClassifierType.CASCADE_CONV_REC and feature_type is FeatureType.SPATIAL_TEMPORAL:
        valid = True
    elif classifier_type in [ClassifierType.DENSE_NET_121, ClassifierType.DENSE_NET_169, ClassifierType.DENSE_NET_201,
                             ClassifierType.VGG16, ClassifierType.VGG19] and \
            feature_type in [FeatureType.MULTI_AVG_FFT_POW, FeatureType.FFT_RANGE, FeatureType.SPATIAL_AVG_FFT_POW]:
        valid = True
    elif classifier_type is ClassifierType.KOLCS_NET and FeatureType.FFT_POWER:
        valid = True
    else:
        valid = False
    assert valid, 'Feature {} is not implemented for classifier {}'.format(feature_type.name, classifier_type.name)


"""Helpers for memory management"""


def __wrapper_func(func, queue, *args):
    ans = dict(result=None, exception=None)
    try:
        ans['result'] = func(*args)
    except Exception as e:
        ans['exception'] = e
    queue.put(ans)


def _process_run(func, *args):
    """Helper function for memory usage management"""
    queue = Queue()
    p = Process(target=__wrapper_func, args=(func, queue) + args)
    p.start()
    ans = queue.get()
    p.join()
    if ans['exception'] is not None:
        print('The error came from {}'.format(func))
        raise ans['exception']
    return ans['result']


class BCISystem(object):
    """Main class for Brain-Computer Interface application.

    This is the main class for the BCI application. Online and offline data manipulation is
    also available.
    """

    def __init__(self, make_logs=False, verbose=True, log_file='log.csv'):
        """Constructor for BCI system.

        Parameters
        ----------
        make_logs : bool
            To make log files or not.
        """
        self._proc = OfflineDataPreprocessor()
        self._log = False
        self._log_file_name = log_file

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
    def _train_classifier(train, validation, classifier_type, input_shape, output_classes, label_encoder,
                          make_binary_classification, batch_size=32, epochs=1, **classifier_kwargs):

        classifier = init_classifier(classifier_type, input_shape, output_classes,
                                     **classifier_kwargs)

        train_ds = DataHandler(train, label_encoder, make_binary_classification).get_tf_dataset()

        if classifier_type == ClassifierType.SVM:
            train_x, train_y = zip(*train_ds.as_numpy_iterator())
            train_x = np.array(train_x)
            train_y = np.array(train_y)
            classifier.fit(train_x, train_y)
        else:
            assert batch_size is not None, 'batch_size can not be None in case of neural networks'
            train_ds = train_ds.batch(batch_size).prefetch(tf_data.experimental.AUTOTUNE)
            if validation is not None:
                val_ds = DataHandler(validation, label_encoder, make_binary_classification). \
                    get_tf_dataset().batch(batch_size)
                classifier.fit(train_ds, validation_data=val_ds, epochs=epochs)
            else:
                classifier.fit(train_ds, epochs=epochs)
        return classifier

    def _one_offline_step(self, train, val, test, classifier_type, labels, classifier_kwargs, label_encoder,
                          make_binary_classification):
        t = time.time()
        if self._verbose:
            print('Training...')

        classifier = self._train_classifier(train, val, classifier_type, self._proc.get_feature_shape(),
                                            len(labels), label_encoder,
                                            make_binary_classification, **classifier_kwargs)

        t = time.time() - t
        if self._verbose:
            print("Training elapsed {} seconds.".format(int(t)))

        test_ds = DataHandler(test, label_encoder, make_binary_classification).get_tf_dataset()
        test_x, test_y = zip(*test_ds.as_numpy_iterator())
        test_x = np.array(test_x)
        test_y = np.array(test_y)

        if classifier_type != ClassifierType.SVM:
            # test_ds = test_ds.batch(batch_size).prefetch(tf_data.experimental.AUTOTUNE)
            classifier.evaluate(test_x, test_y)

        y_pred = classifier.predict(test_x)
        y_pred = label_encoder.inverse_transform(y_pred)
        test_y = label_encoder.inverse_transform(test_y)

        # https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-and-f-measures
        class_report = classification_report(test_y, y_pred)
        conf_matrix = confusion_matrix(test_y, y_pred)
        acc = accuracy_score(test_y, y_pred)

        return class_report, conf_matrix, acc

    def offline_processing(self, db_name=Databases.PHYSIONET, db_config_ver=-1,
                           subject_handle=SubjectHandle.INDEPENDENT_DAYS,
                           feature_params=None,
                           epoch_tmin=0, epoch_tmax=4,
                           window_length=1.0, window_step=.1,
                           filter_params=None,
                           method=XvalidateMethod.SUBJECT,
                           subject_list=None, use_drop_subject_list=True, fast_load=True,
                           subj_n_fold_num=None, shuffle_data=True,
                           make_binary_classification=False, train_file=None,
                           classifier_type=ClassifierType.SVM, classifier_kwargs=None,
                           validation_split=0, do_artefact_rejection=False,
                           make_channel_selection=False,
                           mimic_online_method=False):
        """Offline data processing.

        This method creates an offline BCI-System which make the data preprocessing
        and calculates the classification results.

        Parameters
        ----------
        db_name : Databases
            The database which will be used.
        db_config_ver : float
            Configuration version number of used database. Default is -1
            which defines the latest version.
        subject_handle : SubjectHandle
            Type of subject data loading.
            - INDEPENDENT_DAYS: Handle each experiment as an individual subject.
            - MIX_EXPERIMENTS: Train on all experiments of a given subject.
            - BCI_COMP: BCI competition setup, train and test sets are given.
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
        filter_params : dict, optional
            Parameters for Butterworth highpass digital filtering. ''order'' and ''l_freq''
        method : XvalidateMethod
            The type of cross-validation
        subject_list : int or list of int or None
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
        validation_split : float
            How much of the train set should be used as validation data. value range: [0, 1]
        do_artefact_rejection : bool
            To do artefact-rejection preprocessing or no
        make_channel_selection : bool
            To do channel selection or not.
        mimic_online_method : bool
            If True artefact filtering and channel selection algorithms will be tested
            in online fashion. The parameters of the algorithm will be set on the
            train data and used for the test data. On the other hand, it will generate
            subj_n_fold_num times more processed data, because they can not be reused
            for cross-validation.
        """
        if filter_params is None:
            filter_params = {}
        if classifier_kwargs is None:
            classifier_kwargs = {}
        assert feature_params is not None, 'Feature parameters must be defined.'
        _validate_feature_classifier_pair(feature_params['feature_type'], classifier_type)

        if db_name == Databases.GAME_PAR_D:
            make_binary_classification = True

        select_eeg_file = False
        if train_file is str:
            select_eeg_file = True
            subject_list = 0

        if method == XvalidateMethod.CROSS_SUBJECT:
            cross_subject = True
        elif method == XvalidateMethod.SUBJECT:
            cross_subject = False
            if subject_list is None:
                subject_list = [1]
        else:
            raise NotImplementedError('Method {} is not implemented'.format(method))

        if type(subject_list) is int:
            subject_list = [subject_list]
        print('{}, {} - Subjects: {}'.format(db_name.name, feature_params.get('feature_type').name, subject_list))

        if mimic_online_method:
            subj_n_fold_num = 5 if subj_n_fold_num is None else subj_n_fold_num
            self._proc = OnlineDataPreprocessor(epoch_tmin, epoch_tmax, window_length, window_step,
                                                use_drop_subject_list=use_drop_subject_list, fast_load=fast_load,
                                                filter_params=filter_params,
                                                do_artefact_rejection=do_artefact_rejection,
                                                n_fold=subj_n_fold_num, shuffle=shuffle_data,
                                                make_channel_selection=make_channel_selection,
                                                subject_handle=subject_handle)
        else:
            self._proc = OfflineDataPreprocessor(epoch_tmin, epoch_tmax, window_length, window_step,
                                                 use_drop_subject_list=use_drop_subject_list, fast_load=fast_load,
                                                 select_eeg_file=select_eeg_file,
                                                 eeg_file=train_file, filter_params=filter_params,
                                                 do_artefact_rejection=do_artefact_rejection,
                                                 make_channel_selection=make_channel_selection,
                                                 subject_handle=subject_handle)
        self._proc.use_db(db_name, db_config_ver)
        if make_binary_classification:
            self._proc.validate_make_binary_classification_use()

        def skipp_subject(subject):
            if subject is not None and self._proc.is_subject_in_drop_list(subject):
                print('Subject{} is in the drop list. Can not process and classify data.'.format(subject))
                return True
            return False

        if subject_list is not None and len(subject_list) == 1 and skipp_subject(subject_list[0]):
            return

        if subject_list is not None:
            proc_subjects = list(set(np.array(subject_list).flatten()))
            loader = DataLoader(subject_handle=subject_handle).use_db(db_name, db_config_ver)
            if cross_subject and any(loader.is_subject_in_drop_list(subj) for subj in proc_subjects):
                proc_subjects = None
        else:
            proc_subjects = None

        self._proc.run(proc_subjects, **feature_params)
        assert len(self._proc.get_processed_subjects()) > 0, 'There are no preprocessed subjects...'

        kfold = SubjectKFold(
            self._proc, subj_n_fold_num, validation_split=validation_split, shuffle_data=shuffle_data,
            binarize_db=(db_name != Databases.GAME_PAR_D and make_binary_classification)
        )

        labels = self._proc.get_labels(make_binary_classification)
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)

        if subject_list is None:
            subject_list = [None]

        for subject in subject_list:
            if skipp_subject(subject):
                continue

            cross_acc = list()
            if self._log:  # must be here because of the subject number...
                self._df_base_data = [
                    db_name.name, method, feature_params.get('feature_type').name, subject,
                    epoch_tmin, epoch_tmax, window_length, window_step,
                    feature_params.get('fft_low'), feature_params.get('fft_high'),
                    feature_params.get('fft_step'), feature_params.get('fft_ranges'),
                    classifier_kwargs.get('C'), classifier_kwargs.get('gamma')
                ]

            for train, test, val, subj in kfold.split(subject, cross_subject=cross_subject):
                if classifier_type is ClassifierType.SVM or not tf_test.is_built_with_gpu_support():
                    class_report, conf_matrix, acc = self._one_offline_step(
                        train, val, test, classifier_type, labels, classifier_kwargs,
                        label_encoder, make_binary_classification
                    )
                else:
                    class_report, conf_matrix, acc = _process_run(
                        self._one_offline_step, train, val, test, classifier_type, labels, classifier_kwargs,
                        label_encoder, make_binary_classification
                    )
                cross_acc.append(acc)

                self._log_and_print("####### Classification report for subject{}: #######".format(subj))
                self._log_and_print("classifier %s:\n%s" % (self, class_report))
                self._log_and_print("Confusion matrix:\n%s\n" % conf_matrix)
                self._log_and_print("Accuracy score: {}\n".format(acc))

            self._log_and_print("Avg accuracy: {}".format(np.mean(cross_acc)))
            self._log_and_print("Accuracy scores for k-fold crossvalidation: {}\n".format(cross_acc))
            self._save_params((cross_acc, np.mean(cross_acc)))
            self.log_results(self._log_file_name)

    def play_game(self, db_name=Databases.GAME, feature_params=None,
                  epoch_tmin=0, epoch_tmax=4,
                  window_length=1, pretrain_window_step=0.1,
                  filter_params=None,
                  command_delay=0.5,
                  make_binary_classification=False,
                  use_binary_game_logger=False,
                  make_opponents=False,
                  train_file=None,
                  classifier_type=ClassifierType.SVM,
                  classifier_kwargs=None,
                  batch_size=None,
                  time_out=None,
                  do_artefact_rejection=False):
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
        pretrain_window_step : float
            Step of sliding window in seconds.
        filter_params : dict, optional
            Parameters for Butterworth highpass digital filtering. ''order'' and ''l_freq''
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
        do_artefact_rejection:
            If True, FASTER artefact rejection is performed before processing the signal
        """

        if filter_params is None:
            filter_params = {}
        if classifier_kwargs is None:
            classifier_kwargs = {}
        assert feature_params is not None, 'Feature parameters must be defined.'
        _validate_feature_classifier_pair(feature_params['feature_type'], classifier_type)

        if db_name == Databases.GAME_PAR_D:
            make_binary_classification = True

        self._proc = OfflineDataPreprocessor(epoch_tmin, epoch_tmax,
                                             window_length, pretrain_window_step,
                                             fast_load=False, select_eeg_file=True, eeg_file=train_file,
                                             filter_params=filter_params, do_artefact_rejection=do_artefact_rejection)

        self._proc.use_db(db_name)
        if make_binary_classification:
            self._proc.validate_make_binary_classification_use()
        self._proc.run(**feature_params)
        print('Training...')
        t = time.time()

        feature_extractor = FeatureExtractor(fs=self._proc.fs, info=self._proc.info, **feature_params)

        data_list = self._proc.get_processed_db_source(0, only_files=True)
        labels = self._proc.get_labels(make_binary_classification)

        label_encoder = LabelEncoder()
        label_encoder.fit(labels)

        classifier = self._train_classifier(data_list, None, classifier_type, self._proc.get_feature_shape(),
                                            len(labels), label_encoder, make_binary_classification,
                                            batch_size, **classifier_kwargs)

        print("Training elapsed {} seconds.".format(int(time.time() - t)))

        game_log = None
        if use_binary_game_logger and is_platform('windows'):
            from brainvision import RemoteControlClient
            rcc = RemoteControlClient(print_received_messages=False)
            rcc.open_recorder()
            rcc.check_impedance()
            game_log = GameLogger(bv_rcc=rcc)
            game_log.start()

        if make_opponents:
            create_opponents(main_player=1, game_logger=game_log, reaction=command_delay)

        dsp = online.DSP(use_filter=len(filter_params) > 0, **filter_params)
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

                if do_artefact_rejection:
                    eeg = self._proc.artefact_filter.online_filter(eeg)

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
    feature_extraction = dict(
        feature_type=FeatureType.AVG_FFT_POWER,
        fft_low=14, fft_high=30
    )
    filter_params = dict(  # required for FASTER artefact filter
        order=5, l_freq=1, h_freq=45
    )
    classifier_type = ClassifierType.SVM
    classifier_kwargs = dict(
        # weights=None,
        # batch_size=32,
        # epochs=10,
    )

    bci = BCISystem()
    bci.offline_processing(
        db_name=Databases.TTK,
        feature_params=feature_extraction,
        fast_load=False,
        epoch_tmin=0, epoch_tmax=4,
        window_length=2, window_step=.1,
        method=XvalidateMethod.SUBJECT,
        subject_list=[2, 3],
        use_drop_subject_list=True,
        subj_n_fold_num=5,
        filter_params=filter_params,
        classifier_type=classifier_type,
        classifier_kwargs=classifier_kwargs,
        validation_split=0,
        mimic_online_method=False,
        do_artefact_rejection=False,
        make_channel_selection=False
    )

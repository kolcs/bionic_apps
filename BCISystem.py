import time
from warnings import warn, simplefilter

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import ai
import online
from config import REST
from logger import setup_logger, log_info
from preprocess import OfflineDataPreprocessor, SubjectKFold, save_pickle_data, load_pickle_data, \
    init_base_config, calculate_fft_power, calculate_fft_range, \
    FFT_RANGE, AVG_COLUMN, FFT_POWER

AI_MODEL = 'ai.model'
LOGGER_NAME = 'BCISystem'

# db selection options
PHYSIONET = 'physionet'
PILOT_PAR_A = 'pilot_par_a'
PILOT_PAR_B = 'pilot_par_b'
TTK = 'ttk'
GAME = 'game'
GAME_PAR_C = 'game_par_c'
GAME_PAR_D = 'game_par_d'

# method selection options
CROSS_SUBJECT_X_VALIDATE = 'crossSubjectXvalidate'
SUBJECT_X_VALIDATE = 'subjectXvalidate'
CROSS_SUBJECT_X_AND_SAVE_SVM = 'crossSubXandTrainSVM'

# LOG, pandas columns
LOG_COLS = ['Database', 'Method', 'Feature', 'Subject', 'Epoch tmin', 'Epoch tmax', 'Window length', 'Window step',
            'FFT low', 'FFT high', 'FFT step', 'Accuracy list', 'Avg. Acc']


# LOG_DB_NAME = 'Database'
# LOG_METHOD = 'Method'
# LOG_FEATURE = 'Feature'
# LOG_SUBJECT = 'Subject'
# LOG_EPOCH_TMIN = 'Epoch tmin'
# LOG_EPOCH_TMAX = 'Epoch tmax'
# LOG_WINDOW_LENGTH = 'Window length'
# LOG_WINDOW_STEP = 'Window step'
# LOG_FFT_LOW = 'FFT low'
# LOG_FFT_HIGH = 'FFT high'
# LOG_FFT_STEP = 'FFT step'

class BCISystem(object):

    def __init__(self, feature=FFT_POWER, window_length=0.5, window_step=0.25, make_logs=False):
        """ Constructor for BCI system

        Parameters
        ----------
        feature : {'avg_column', 'fft_power', 'fft_range'}
            The feature which will be created.
        window_length: float
            length of eeg processor window in seconds
        window_step: float
            window shift in seconds
        make_logs : bool
            To make log files or not.
        """
        self._base_dir = init_base_config()
        self._window_length = window_length
        self._window_step = window_step
        self._proc = None
        self._prev_timestamp = [0]
        self._ai_model = dict()
        self._log = make_logs
        self._feature = feature

        self._df = None
        self._df_base_data = list()

        if make_logs:
            setup_logger(LOGGER_NAME)
            self._df = pd.DataFrame(columns=LOG_COLS)

    def _log_and_print(self, msg):
        if self._log:
            log_info(LOGGER_NAME, msg)
        print(msg)

    def _save_params(self, args):  # implemeted for _subject_crossvalidate()
        if self._log:
            data = self._df_base_data.copy()
            data.extend(args)
            s = pd.Series(data, index=LOG_COLS)
            self._df = self._df.append(s, ignore_index=True)

    def show_results(self, out_file_name='out.csv'):
        if self._log:
            print(self._df)
            self._df.to_csv(out_file_name, sep=';', encoding='utf-8', index=False)

    def _init_svm(self, C=1, cache_size=2000, random_state=None):
        if self._feature == FFT_RANGE:
            return ai.MultiSVM(C=C, cache_size=cache_size, random_state=random_state)
        return ai.SVM(C=C, cache_size=cache_size, random_state=random_state)

    def _subject_corssvalidate(self, subject=None, k_fold_num=10):
        if subject is None:
            subject = 1
        kfold = SubjectKFold(k_fold_num)

        self._log_and_print("####### Classification report for subject{}: #######".format(subject))
        cross_acc = list()

        for train_x, train_y, test_x, test_y in kfold.split_subject_data(self._proc, subject):
            t = time.time()
            print('Training...')

            svm = self._init_svm(C=1)
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
        self._save_params((cross_acc, np.mean(cross_acc)))  # todo: print result...

    def _crosssubject_crossvalidate(self, subj_n_fold_num=None, save_model=False):
        kfold = SubjectKFold(subj_n_fold_num)

        for train_x, train_y, test_x, test_y, test_subject in kfold.split(self._proc):
            t = time.time()
            print('Training...')

            svm = self._init_svm(C=1)  # , class_weight='balanced')
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

    def _init_db_processor(self, db_name, epoch_tmin=0, epoch_tmax=3, window_lenght=None, window_step=None,
                           use_drop_subject_list=True, fast_load=True, make_binary_classification=False,
                           subject=None, play_game=False):
        """Database initializer.

        Initialize the database preprocessor for the required db, which handles the configuration.

        Parameters
        ----------
         db_name : {'physionet', 'pilot_parA', 'pilot_parB', 'ttk'}
            Database to work on...
        epoch_tmin : int
            Defining epoch start from trigger in seconds.
        epoch_tmax : int
            Defining epoch end from trigger in seconds.
        use_drop_subject_list : bool
            Whether to use drop subject list from config file or not?
        fast_load : bool
            Handle with extreme care! It loads the result of a previous preprocess task.
        subject : int or list of int
            Data preprocess is made on these subjects.
        """
        if self._proc is None:
            if window_lenght is not None:
                self._window_length = window_lenght
            if window_step is not None:
                self._window_step = window_step

            self._proc = OfflineDataPreprocessor(self._base_dir, epoch_tmin, epoch_tmax, use_drop_subject_list,
                                                 self._window_length, self._window_step, fast_load,
                                                 make_binary_classification, subject, play_game)
            if db_name == PHYSIONET:
                self._proc.use_physionet()
                # labels = [REST, LEFT_HAND, RIGHT_HAND, BOTH_LEGS, BOTH_HANDS]
            elif db_name == PILOT_PAR_A:
                self._proc.use_pilot_par_a()
                # labels = [REST, LEFT_HAND, RIGHT_HAND, BOTH_LEGS, BOTH_HANDS]
            elif db_name == PILOT_PAR_B:
                self._proc.use_pilot_par_b()
            elif db_name == TTK:
                self._proc.use_ttk_db()
            elif db_name == GAME:
                self._proc.use_game_data()
            elif db_name == GAME_PAR_C:
                self._proc.use_game_par_c()
            elif db_name == GAME_PAR_D:
                self._proc.use_game_par_d()

            else:
                raise NotImplementedError('Database processor for {} db is not implemented'.format(db_name))

    def clear_db_processor(self):
        self._proc = None

    def _feature_extraction(self, data, fft_low=7, fft_high=13, fs=500):

        if self._feature == AVG_COLUMN:
            data = np.average(data, axis=-1)
            data = data.reshape((1, -1))
        elif self._feature == FFT_POWER:
            data = calculate_fft_power(data, fs, fft_low, fft_high)
        elif self._feature == FFT_RANGE:
            data = calculate_fft_range(data, fs, fft_low, fft_high)
        else:
            raise NotImplementedError('{} feature is not implemented'.format(self._feature))

        return data

    def offline_processing(self, db_name=PHYSIONET, feature=None, fft_low=7, fft_high=13, fft_step=2,
                           fft_width=2, method=CROSS_SUBJECT_X_VALIDATE, epoch_tmin=0, epoch_tmax=3, window_length=0.5,
                           window_step=0.25, subject=None, use_drop_subject_list=True, fast_load=False,
                           subj_n_fold_num=None, make_binary_classification=False):
        if feature is not None:
            self._feature = feature
        if window_length is not None:
            self._window_length = window_length
        if window_step is not None:
            self._window_step = window_step
        self._init_db_processor(db_name, epoch_tmin, epoch_tmax, self._window_length, self._window_step,
                                use_drop_subject_list, fast_load, make_binary_classification, subject)
        self._proc.run(self._feature, fft_low, fft_high, fft_step, fft_width)

        if self._log:
            self._df_base_data = [db_name, method, self._feature, subject, epoch_tmin, epoch_tmax, self._window_length,
                                  self._window_step, fft_low, fft_high, fft_step]

        if method == CROSS_SUBJECT_X_VALIDATE:
            self._crosssubject_crossvalidate(subj_n_fold_num)

        elif method == CROSS_SUBJECT_X_AND_SAVE_SVM:
            self._crosssubject_crossvalidate(save_model=True)

        elif method == SUBJECT_X_VALIDATE:
            self._subject_corssvalidate(subject, subj_n_fold_num)

        else:
            raise NotImplementedError('Method {} is not implemented'.format(method))

    def online_processing(self, db_name, test_subject, feature=None, get_real_labels=False, data_sender=None):
        """Online accuracy check.

        This is an example code, how the online classification can be done.

        Parameters
        ----------
        db_name : {'physionet', 'pilot_parA', 'pilot_parB', 'ttk'}
            Database to work on...
        test_subject : int
            Test subject number, which was not included in ai training.
        feature : {'avg_column'}, optional
            Feature created from EEG window
        get_real_labels : bool, optional
            Load real labels from file to test accuracy.
        data_sender : multiprocess.Process, optional
            Process object which sends the signals for simulating realtime work.
        """
        if feature is not None:
            self._feature = feature
        self._init_db_processor(db_name)
        self._proc.init_processed_db_path(self._feature)
        if len(self._ai_model) == 0:
            self._ai_model = load_pickle_data(self._proc.proc_db_path + AI_MODEL)
        svm = self._ai_model[test_subject]
        self._ai_model = None
        dsp = online.DSP()
        # dsp.start_parallel_signal_recording(rec_type='chunk')  # todo: have it or not?
        sleep_time = 1 / dsp.fs

        y_preds = list()
        y_real = list()
        label = None
        drop_count = 0
        dstim = {1: 0, 5: 0, 7: 0, 1001: 0, 9: 0, 11: 0, 12: 0, 15: 0}

        while data_sender is None or data_sender.is_alive():
            # t = time.time()

            if get_real_labels:
                timestamps, data, label = dsp.get_eeg_window(self._window_length, get_real_labels)
                # dstim[label] += 1
            else:
                timestamps, data = dsp.get_eeg_window(self._window_length)

            sh = np.shape(data)
            if len(sh) < 2 or sh[1] / dsp.fs < self._window_length or timestamps == self._prev_timestamp:
                # The data is still not big enough for window.
                # time.sleep(1/dsp.fs)
                drop_count += 1
                continue

            # print('Dropped: {}\n time diff: {}'.format(drop_count, (timestamps[0]-self._prev_timestamp[0])*dsp.fs))
            drop_count = 0

            self._prev_timestamp = timestamps

            data = self._feature_extraction(data, dsp.fs)

            y_pred = svm.predict(data)

            y_real.append(label)
            y_preds.append(y_pred)
            # time.sleep(max(0, sleep_time - (time.time() - t)))  # todo: Do not use - not real time...

        raw = np.array(dsp._eeg)
        stims = raw[:, -1]
        # check_received_signal(raw[:, :-1], file)

        for s in stims:
            dstim[s] += 1
        print('received stim', dstim)

        return y_preds, y_real, raw

    def play_game(self, db_name=GAME, feature=None, fft_low=7, fft_high=13, epoch_tmin=0, epoch_tmax=3,
                  window_length=None, window_step=None, command_in_each_sec=0.5, make_binary_classification=False):
        if feature is not None:
            self._feature = feature
        if window_length is not None:
            self._window_length = window_length
        if window_step is not None:
            self._window_step = window_step
        if db_name == GAME_PAR_D:
            make_binary_classification = True

        self._init_db_processor(db_name=db_name, epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
                                window_lenght=self._window_length, window_step=self._window_step,
                                use_drop_subject_list=False, fast_load=False,
                                make_binary_classification=make_binary_classification,
                                play_game=True)
        self._proc.run(self._feature, fft_low, fft_high)
        print('Training...')
        t = time.time()
        data, labels = self._proc.get_subject_data(0)
        svm = self._init_svm(C=1)
        svm.fit(data, labels)
        print("Training elapsed {} seconds.".format(int(time.time() - t)))

        dsp = online.DSP()

        from control import GameControl
        controller = GameControl(make_log=True, log_to_stream=True)
        command_converter = self._proc.get_command_converter() if not make_binary_classification else dict()

        print("Starting game control...")
        simplefilter('always', UserWarning)
        while True:
            timestamp, eeg = dsp.get_eeg_window_in_chunk(self._window_length)
            if timestamp is not None:
                tic = time.time()
                eeg = np.delete(eeg, -1, axis=0)  # removing last unwanted channel
                data = self._feature_extraction(eeg, fft_low, fft_high, fs=dsp.fs)
                y_pred = svm.predict(data)[0]

                if make_binary_classification:
                    controller.control_game_with_2_opt(y_pred)
                else:
                    command = command_converter[y_pred]
                    controller.control_game(command)

                toc = time.time() - tic
                if toc < command_in_each_sec:
                    time.sleep(command_in_each_sec - toc)
                else:
                    warn('Classification took longer than command giving limit!')


def check_received_signal(data, filename):
    from preprocess import open_raw_file
    raw = open_raw_file(filename)
    info = raw.info
    from online.DataSender import get_data_with_labels
    _, _, orig = get_data_with_labels(raw)
    orig.plot(title='Sent')
    from mne.io import RawArray
    raw = RawArray(np.transpose(data), info)
    raw.plot(title='Received')
    from matplotlib import pyplot as plt
    plt.show()
    print(orig.get_data().shape, raw.get_data().shape)
    d_orig = orig.get_data()
    d_raw = raw.get_data()
    for t in range(len(raw)):
        print('orig: {}\nreceived: {}\nmatch {}'.format(d_orig[:, t], d_raw[:, t], d_orig[:, t] == d_raw[:, t]))


def calc_online_acc(y_pred, y_real, raw):
    save_pickle_data(y_real, 'tmp/y_real.data')
    save_pickle_data(y_pred, 'tmp/y_pred.data')
    save_pickle_data(raw, 'tmp/eeg.data')
    from config import PilotDB_ParadigmA
    conv = {val: key for key, val in PilotDB_ParadigmA.TRIGGER_TASK_CONVERTER.items()}
    y_real = [conv.get(y, REST) for y in y_real]
    print('\nDiff labels: {}\n'.format(set(np.array(y_pred).flatten())))
    class_report = classification_report(y_real, y_pred)
    conf_martix = confusion_matrix(y_real, y_pred)
    acc = accuracy_score(y_real, y_pred)

    print("%s\n" % class_report)
    print("Confusion matrix:\n%s\n" % conf_martix)
    print("Accuracy score: {}\n".format(acc))


if __name__ == '__main__':
    bci = BCISystem()
    bci.offline_processing(db_name=PILOT_PAR_B, feature=FFT_POWER, method=CROSS_SUBJECT_X_VALIDATE)

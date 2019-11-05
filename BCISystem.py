import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import ai
import online
from config import *
from logger import *
from preprocess import OfflineDataPreprocessor, SubjectKFold, save_pickle_data, load_pickle_data, calculate_fft_power, \
    FFT_RANGE, AVG_COLUMN, FFT_POWER

AI_MODEL = 'ai.model'
LOGGER_NAME = 'BCISystem'

CONFIG_FILE = 'bci_system.cfg'
# config options
BASE_DIR = 'base_dir'


class BCISystem(object):

    def __init__(self, window_length=0.5, window_step=0.25, make_logs=False):
        """ Constructor for BCI system

        Parameters
        ----------
        window_length: float
            length of eeg processor window in seconds
        window_step: float
            window shift in seconds
        """
        self._base_dir = str()
        self._window_length = window_length
        self._window_step = window_step
        self._proc = None
        self._prev_timestamp = [0]
        self._ai_model = dict()
        self._log = make_logs

        self._init_base_config()

        if make_logs:
            setup_logger(LOGGER_NAME)

    def _init_base_config(self):
        cfg_dict = load_pickle_data(CONFIG_FILE)
        if cfg_dict is None:
            from gui_handler import select_base_dir
            base_dir = select_base_dir()
            assert base_dir is not None, 'Base directory not selected. Cannot run program!'
            self._base_dir = base_dir
            cfg_dict = {BASE_DIR: base_dir}
            save_pickle_data(cfg_dict, CONFIG_FILE)
        else:
            self._base_dir = cfg_dict[BASE_DIR]

    def _log_and_print(self, msg):
        if self._log:
            log_info(LOGGER_NAME, msg)
        print(msg)

    def _subject_corssvalidate(self, subject, k_fold_num=10, feature=None):
        kfold = SubjectKFold(k_fold_num)

        self._log_and_print("####### Classification report for subject{}: #######".format(subject))
        cross_acc = list()

        for train_x, train_y, test_x, test_y in kfold.split_subject_data(self._proc, subject):
            t = time.time()
            print('Training...')

            if feature == FFT_RANGE:
                svm = ai.MultiSVM(C=1, cache_size=4000, random_state=12)
            else:
                svm = ai.SVM(C=1, cache_size=4000, random_state=12)
            svm.fit(train_x, train_y)

            t = time.time() - t
            print("Training elapsed {} seconds.".format(int(t)))

            y_pred = svm.predict(test_x)

            class_report = classification_report(test_y, y_pred)
            conf_martix = confusion_matrix(test_y, y_pred)
            acc = accuracy_score(test_y, y_pred)
            cross_acc.append(acc)

            self._log_and_print("classifier %s:\n%s" % (self, class_report))
            self._log_and_print("Confusion matrix:\n%s\n" % conf_martix)
            self._log_and_print("Accuracy score: {}\n".format(acc))

        self._log_and_print("Avg accuracy: {}".format(np.mean(cross_acc)))
        self._log_and_print("Accuracy scores for k-fold crossvalidation: {}\n".format(cross_acc))

    def _crosssubject_crossvalidate(self, subj_n_fold_num=None, save_model=False):
        kfold = SubjectKFold(subj_n_fold_num)

        for train_x, train_y, test_x, test_y, test_subject in kfold.split(self._proc):
            t = time.time()
            print('Training...')
            svm = ai.SVM(C=1, cache_size=4000, random_state=12)  # , class_weight='balanced')

            svm.fit(train_x, train_y)
            t = time.time() - t
            print("Training elapsed {} seconds.".format(int(t)))

            y_pred = svm.predict(test_x)

            class_report = classification_report(test_y, y_pred)
            conf_martix = confusion_matrix(test_y, y_pred)
            acc = accuracy_score(test_y, y_pred)

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
                           use_drop_subject_list=True, fast_load=True):
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
        """
        if self._proc is None:
            if window_lenght is not None:
                self._window_length = window_lenght
            if window_step is not None:
                self._window_step = window_step

            self._proc = OfflineDataPreprocessor(self._base_dir, epoch_tmin, epoch_tmax, use_drop_subject_list,
                                                 self._window_length, self._window_step, fast_load)
            if db_name == 'physionet':
                self._proc.use_physionet()
                # labels = [REST, LEFT_HAND, RIGHT_HAND, BOTH_LEGS, BOTH_HANDS]
            elif db_name == 'pilot_parA':
                self._proc.use_pilot()
                # labels = [REST, LEFT_HAND, RIGHT_HAND, BOTH_LEGS, BOTH_HANDS]
            elif db_name == 'pilot_parB':
                self._proc.use_pilot_par_b()
            elif db_name == 'ttk':
                self._proc.use_ttk_db()
            elif db_name == 'game':
                self._proc.use_game_data()

            else:
                raise NotImplementedError('Database processor for {} db is not implemented'.format(db_name))

    def _feature_extraction(self, data, feature='fft_power', fft_low=7, fft_high=13, fs=500):

        if feature == AVG_COLUMN:
            data = np.average(data, axis=-1)
        elif feature == FFT_POWER:
            data = calculate_fft_power(data, fs, fft_low, fft_high)
        else:
            raise NotImplementedError('{} feature creation is not implemented'.format(feature))

        # Do this only for svm data!!!
        data = data.reshape((1, -1))
        return data

    def offline_processing(self, db_name='physionet', feature=FFT_POWER, fft_low=7, fft_high=13, fft_step=2,
                           fft_width=2, method='crossSubjectXvalidate', epoch_tmin=0, epoch_tmax=3, window_length=0.5,
                           window_step=0.25, subject=1, use_drop_subject_list=True, fast_load=False,
                           subj_n_fold_num=None):

        self._init_db_processor(db_name, epoch_tmin, epoch_tmax, window_length, window_step, use_drop_subject_list,
                                fast_load)
        self._proc.run(feature, fft_low, fft_high, fft_step, fft_width)

        if method == 'crossSubjectXvalidate':
            self._crosssubject_crossvalidate(subj_n_fold_num)

        elif method == 'trainSVM':
            self._crosssubject_crossvalidate(save_model=True)

        elif method == 'subjectXvalidate':
            self._subject_corssvalidate(subject, subj_n_fold_num, feature)

        else:
            raise NotImplementedError('Method {} is not implemented'.format(method))

    def online_processing(self, db_name, test_subject, feature='avg_column', get_real_labels=False, data_sender=None):
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
        self._init_db_processor(db_name)
        self._proc.init_processed_db_path(feature)
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

            data = self._feature_extraction(data, feature)

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

    def play_game(self, feature='fft_power', fft_low=7, fft_high=13, epoch_tmin=0, epoch_tmax=3, window_length=0.5,
                  window_step=0.25):
        self._init_db_processor('game', epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax, window_lenght=window_length,
                                window_step=window_step, use_drop_subject_list=False, fast_load=False)
        self._proc.run(feature, fft_low, fft_high)

        print('Training...')
        t = time.time()
        data, labels = self._proc.get_subject_data(0)
        svm = ai.SVM(C=1, cache_size=4000)
        svm.fit(data, labels)
        print("Training elapsed {} seconds.".format(int(time.time() - t)))

        dsp = online.DSP()

        from control import GameControl
        controller = GameControl(make_log=True)
        command_converter = self._proc.get_command_converter()

        print("Starting game control...")
        while True:
            timestamp, eeg = dsp.get_eeg_window_in_chunk(window_length)
            if timestamp is not None:
                data = self._feature_extraction(eeg, feature='fft_power', fs=dsp.fs)
                y_pred = svm.predict(data)
                command = command_converter[y_pred]
                controller.control_game(command)
                time.sleep(0.1)


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
    from config import PilotDB
    conv = {val: key for key, val in PilotDB.TRIGGER_TASK_CONVERTER.items()}
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
    bci.offline_processing(db_name='pilot_parB', feature='fft_power', method='crossSubjectXvalidate')

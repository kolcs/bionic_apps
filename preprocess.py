import pyedflib
import numpy as np
import pickle
import mne
from config import *


class EDFHandler:
    """
    Used for handling edf files. It loads the specified data to the memory only if it is needed.
    Therefore we only use 3.76 MB memory instead of 3.36 GB (in case of physionet) ....
    """

    def __init__(self, filename):
        self._filename = filename
        self._file_handler = None
        self._from = 0
        self._duration = None
        self._used = False

    def _load_file_to_memory(self):
        """
        Opens the corresponding file given in self._filename to self._file_handler

        :raises FileNotFoundError if no filename were specified.
        """
        if self._filename:
            if not self._file_handler:
                self._file_handler = mne.io.read_raw_edf(self._filename, preload=True)
        else:
            raise FileNotFoundError('Filename was not specified for edf file - can not open')

    def close_file(self):
        """
        Resets self._file_handler to None
        """
        self._file_handler.close()
        self._file_handler = None

    def is_used(self):
        return self._used

    def get_frequency(self):
        self._load_file_to_memory()
        return self._file_handler.info['sfreq']

    def get_channels(self):
        self._load_file_to_memory()
        return self._file_handler.info['ch_names']

    def get_basic_infos(self):
        f = self.get_frequency()
        ch = self.get_channels()
        self.close_file()
        return f, ch

    def _get_num_with_predefined_char(self, char, required_num='required'):
        """
        Searches for number in filename

        :param char: character to search for
        :param required_num: string to print in error message
        :return: number after given char
        :raises FileNotFoundError if none numbers were found
        """
        import re
        num_list = re.findall(r'.*' + char + '\d+', self._filename)
        if not num_list:
            raise FileNotFoundError(
                "Can not give back {} number: filename '{}' does not contain '{}' character.".format(required_num,
                                                                                                     self._filename,
                                                                                                     char))
        num_str = num_list[0]
        num_ind = num_str.rfind(char) - len(num_str) + 1
        return int(num_str[num_ind:])

    def get_record_number(self):
        """
        Only works if record number is stored in filename:
            *R<record num>*

        :return: record number from filename
        """
        return self._get_num_with_predefined_char('R', "RECORD")

    def get_subject_number(self):
        """
        Only works if subject number is stored in filename:
            *S<record num>*

        :return: record number from filename
        """
        return self._get_num_with_predefined_char('S', "SUBJECT")

    def set_window(self, from_, duration):
        """
        Set the window parameters to read eeg data

        :param from_: beginning of window in seconds corresponding to the whole data
        :param duration: how long is the window in seconds
        :return: EDFHandler object
        """
        self._from = from_
        self._duration = duration
        return self

    def get_window(self):
        return self._from, self._duration

    def clear_used(self):
        self._used = False

    def get_annotation(self):
        """
        Loads the annotation / triggers from file.

        :return:  list of trigger start, list of trigger duration, trigger ID
        """
        self._load_file_to_memory()
        annotation = np.transpose(self._file_handler.find_edf_events())
        annotation = (annotation[0, :].astype(np.float), annotation[1, :].astype(np.float), annotation[2, :])
        self.close_file()
        return annotation

    def get_data(self, start_after=0, end_before=0):  # TODO: missing data....
        """
        Loads the specified data in the given window.

        :param start_after: how many seconds to drop from the beginning of the data
        :param end_before: how many seconds to drop from the end of the data
        :return: data matrix: row-time column-channels
        """
        self._load_file_to_memory()
        n = self._file_handler.info['nchan']  # trigger channel!
        data, time = self._file_handler[:n, :]
        # channel_num = self._file_handler.signals_in_file  # number of channels
        n_all_samples = np.size(data, 1)
        fs = self.get_frequency()

        shift_beginning = int(start_after * fs)
        from_ = int(self._from * fs) + shift_beginning
        n_samples = -(int(end_before * fs) + shift_beginning)

        if self._duration:
            n_samples += int(self._duration * fs)
        else:
            n_samples += n_all_samples

        assert 0 < n_samples, "The dropping values are bigger than the whole data window."

        sigbufs = data[:, from_:n_samples]
        self._used = True

        self.close_file()

        return sigbufs


"""
Main parameters for Preprocessor
"""

F_EXT = ".pkl"
DB_FILE_NAME = 'preprocessed_database'


class DBProcessor:
    """
    Preprocessor for edf files. Creates a database, which has all the required information about the eeg files.
    """

    def __init__(self, base_dir, fast_load=True, use_drop_subject_list=True):
        self._base_dir = base_dir
        self._data_path = None
        self._data_base = self.DataBase()
        self._perform_fast_load = fast_load
        self._data_duration = 4  # seconds

        self._drop_subject = set()
        if use_drop_subject_list:
            self._drop_subject.add(-1)

        if not base_dir[-1:] == '/':
            self._base_dir = base_dir + '/'

    class DataBase:
        """
        Class for organising the data in one package
        """

        def __init__(self):
            """
            Inner data of self._data_with_subject and self._data_with_no_subject are equivalent!
            """
            self._data_with_subject = {BASELINE: dict(), IMAGINED_MOVEMENT: dict(), REAL_MOVEMENT: dict()}
            self._data_with_no_subject = {BASELINE: dict(), IMAGINED_MOVEMENT: dict(), REAL_MOVEMENT: dict()}
            self.fs = 0
            self.channels = 0

        @property
        def subject(self):
            return self._data_with_subject

        @property
        def no_subject(self):
            return self._data_with_no_subject

        def get_task_by_type_subject(self, task_type):
            """
            Returnes the type of action: imagined / real

            :param task_type: real / imagined / eye
            :return: dict of task_type oriented data
            """
            return self._data_with_subject.get(task_type)

        def get_task_by_type_no_subject(self, task_type):
            """
            Returnes the type of action: imagined / real

            :param task_type: real / imagined / eye
            :return: dict of task_type oriented data
            """
            return self._data_with_no_subject.get(task_type)

        def get_subject(self, task_type, subject_ID):
            """
            Returns subject connected database part

            :param task_type: real / imagined / eye
            :param subject_ID: subject
            :return: dict of subject oriented data
            """
            return self.get_task_by_type_subject(task_type).get(subject_ID)

        def get_trigger_list_subject(self, task_type, subject_ID, trigger_ID):
            """
            Returns list of trigger specified edf file

            :param task_type: real / imagined / eye
            :param subject_ID: subject
            :param trigger_ID: specific trigger command (left hand, right hand...)
            :return: list of specific triggers
            """
            return self.get_subject(task_type, subject_ID).get(trigger_ID)

        def get_trigger_list_no_subject(self, task_type, trigger_ID):
            """
            Returns list of trigger specified edf file

            :param task_type: real / imagined / eye
            :param trigger_ID: specific trigger command (left hand, right hand...)
            :return: list of specific triggers
            """
            return self.get_task_by_type_no_subject(task_type).get(trigger_ID)

    def _use_db(self, path, drop_subjects=None, data_duration=None, decimal_precision=None):
        """
        Loads a specified database.

        :param path: where the database is available
        """
        self._data_path = self._base_dir + path
        if drop_subjects and self._drop_subject:
            self._drop_subject = drop_subjects
        if data_duration:
            self._data_duration = data_duration
        # if decimal_precision:
        #     self._decimal_precision = decimal_precision

    def run(self):
        self._load_database()

    def use_physionet(self):
        self._use_db(Physionet.DIR, Physionet.DROP_SUBJECTS, Physionet.MAX_DURATION)
        return self

    @staticmethod
    def get_filenames_in(path, ext='.edf'):
        """
        Searches for files in the given path with specified extension

        :param path: path where to do the search
        :param ext: file extension to search for
        :return: list of filenames
        :raises FileNotFoundError if no files were found
        """
        import glob
        files = glob.glob(path + '/**/*' + ext, recursive=True)
        if not files:
            raise FileNotFoundError('Database is not available under the given db_dir: {}'.format(path))
        return files

    @staticmethod
    def _conv_type_physionet(record_num):  # todo: uiform command
        return Physionet.TRIGGER_TYPE_CONVERTER.get(record_num)

    @staticmethod
    def _conv_task_physionet(record_num, task_ID):  # todo: uiform command
        return Physionet.TRIGGER_TASK_CONVERTER.get(record_num).get(task_ID)

    """
    Database functions
    """

    def _get_filenames(self):
        return self.get_filenames_in(self._data_path)

    def _load_database(self):
        """
        Function for quick load, if it is possible.
        """
        import os
        path = self._data_path + DB_FILE_NAME + F_EXT
        if self._perform_fast_load and os.path.exists(path):
            self._load_preproc_db()
        elif self._perform_fast_load:
            print(
                "There is no preprocessed database on the given path {}\nTherefore  we are creating it.".format(path))
            self._create_annotated_db()
        else:
            self._create_annotated_db()

    def _create_annotated_db(self, clip_side="back"):
        """
        Make preprocessing tasks for edf fliles.
        Sorts the eeg signals by triggers and orders it to a database.
        """
        # Interactive terminal...
        import time
        from itertools import cycle
        proc_start_time = time.time()
        tic = time.time()
        message = 'Creating database'
        it = cycle([message + i * '.' + (4 - i) * ' ' for i in range(4)])
        print("\r{}".format(next(it)), end='')

        files = self._get_filenames()
        self._data_base.fs, self._data_base.channels = EDFHandler(files[0]).get_basic_infos()

        # for all files
        for file in files:
            edf_file = EDFHandler(file)
            rec_num = edf_file.get_record_number()
            from_, duration, triggers = edf_file.get_annotation()
            diff_triggers = list(set(triggers))

            rec_type = self._conv_type_physionet(rec_num)
            s_num = edf_file.get_subject_number()
            subject = SUBJECT + str(s_num)

            if s_num in self._drop_subject:  # drop subjects with given number
                # print("Dropping {} in record {}\ntriggers: {}, \nfrom: {}\nduration: {}".format(subject, rec_num,
                #                                                                                 triggers, from_,
                #                                                                                 duration))
                continue

            if not self._data_base.get_subject(rec_type, subject):
                self._data_base.get_task_by_type_subject(rec_type).update({subject: dict()})

            # for all different triggers
            for trigger in diff_triggers:
                tgr_inds = [i for i, x in enumerate(triggers) if x == trigger]

                conv_tgr = self._conv_task_physionet(rec_num, trigger)

                if not self._data_base.get_trigger_list_subject(rec_type, subject, conv_tgr):
                    self._data_base.get_subject(rec_type, subject).update({conv_tgr: list()})

                if not self._data_base.get_trigger_list_no_subject(rec_type, conv_tgr):
                    self._data_base.get_task_by_type_no_subject(rec_type).update({conv_tgr: list()})

                if rec_type == BASELINE:  # creating 4 sec long intervals from 60 sec
                    n = int(duration[0] // self._data_duration)
                    tgr_inds = [i for i in range(n)]
                    duration = [self._data_duration for _ in range(n)]
                    from_ = [i * self._data_duration for i in range(n)]

                # save each trigger duration
                for ind in tgr_inds:
                    frm = self._clip_data(from_[ind], duration[ind], side="end")
                    if duration[ind] < self._data_duration:
                        print(
                            "Required duration {} is bigger than {} duration at {} record{} {} - Dropping it..".format(
                                self._data_duration, duration[ind], subject, rec_num, trigger))
                        continue
                    data = EDFHandler(file).set_window(frm, self._data_duration)

                    # # todo: remove
                    # fs, _ = data.get_basic_infos()
                    # if not (duration[ind] == 4.1 or duration[ind] == 4.2 or rec_type == BASELINE) or not fs == 160:
                    #     print(subject, rec_num, rec_type, trigger, duration[ind], fs)

                    db_trigger = self._data_base.get_trigger_list_subject(rec_type, subject, conv_tgr)
                    db_trigger.append(data)
                    db_trigger = self._data_base.get_trigger_list_no_subject(rec_type, conv_tgr)
                    db_trigger.append(data)

            # Interactive terminal...
            if time.time() - tic > 1:
                print("\r{}".format(next(it)), end='')
                tic = time.time()

        # Printing process duration...
        print("\nPreprocessing finished under {} seconds".format(time.time() - proc_start_time))

        self._save_preproc_db()

    def _clip_data(self, from_, duration, side="end"):
        if side == "front":
            clip = duration - self._data_duration
            frm = np.round(from_ + clip, decimals=4)  # 1/1000 second precision
        else:
            frm = from_
        return frm

    def _load_preproc_db(self):
        self._data_base = self._load_data(DB_FILE_NAME)

    def _load_data(self, filename):
        with open(self._data_path + filename + F_EXT, 'rb') as fin:
            data = pickle.load(fin)
        return data

    def _save_preproc_db(self):
        self._save_data(self._data_base, DB_FILE_NAME)

    def _save_data(self, data, filename):
        with open(self._data_path + filename + F_EXT, 'wb') as f:
            pickle.dump(data, f)

    def check_db_subject(self):
        """
        Checker for database...
        """
        for record_type in self._data_base.subject.keys():
            for subject in self._data_base.get_task_by_type_subject(record_type).keys():
                print("{} --> {} -->".format(record_type, subject))
                for task in self._data_base.get_subject(record_type, subject).keys():
                    if task:
                        print("{}: {} records".format(task, len(self._data_base.get_trigger_list_subject(record_type,
                                                                                                         subject,
                                                                                                         task))))
                        print(self._data_base.get_trigger_list_subject(record_type,
                                                                       subject,
                                                                       task)[0].get_data())
                print()

    def check_db_no_subject(self):
        """
        Checker for database...
        """
        for record_type in self._data_base.no_subject.keys():
            for task in self._data_base.get_task_by_type_no_subject(record_type).keys():
                print("{} --> {} -->".format(record_type, task))
                print("{} records\n".format(len(self._data_base.get_trigger_list_no_subject(record_type, task))))

    def get_trigger_list_subject(self, task_type, subject_ID, trigger_ID):
        return self._data_base.get_trigger_list_subject(task_type, subject_ID, trigger_ID)

    def get_trigger_list_no_subject(self, task_type, trigger_ID):
        return self._data_base.get_trigger_list_no_subject(task_type, trigger_ID)

    @staticmethod
    def create_directory_on_path(path):
        import os
        if not os.path.exists(path):
            os.makedirs(path)

    def _process_and_save_data(self, data_handler, record_type, task, directory, use_signal_preprocessor,
                               file_counter=0):
        self.create_directory_on_path(directory)
        # print("data shape: {}, window {}".format(np.shape(handler.get_data()), handler.get_window()))
        data = data_handler.get_data()

        if use_signal_preprocessor:
            # todo: here or in read in? python function can be impl in tf data.map(_parser_func)
            data = self._preprocess_signals(data, self._data_base.fs)

        filename = directory + 'file' + str(file_counter) + F_EXT_TF_RECORD
        self._save_to_tf_record(filename, data, record_type, task)

    def convert_data_to_tfRecords(self, train=.8, test=.1, use_signal_preprocessor=False):
        file_counter = 0
        directory = self._base_dir + DIR_TF_RECORDS
        subject_list = self._data_base.get_task_by_type_subject(BASELINE).keys()
        num_subj = len(subject_list)
        for i, subject in enumerate(subject_list):
            if i < train * num_subj:
                dir = directory + DIR_TRAIN
            elif i < num_subj * (train + test):
                dir = directory + DIR_TEST
            else:
                dir = directory + DIR_VALIDATION

            for record_type in self._data_base.subject.keys():
                for task in self._data_base.get_subject(record_type, subject).keys():
                    print("Creating tf Records for {}: {} --> {}".format(subject, record_type, task))
                    for handler in self._data_base.get_trigger_list_subject(record_type, subject, task):
                        self._process_and_save_data(handler, record_type, task, dir, use_signal_preprocessor,
                                                    file_counter)
                        file_counter += 1
        print("Finished...")

    def convert_data_to_tfRecords_no_subject(self, use_signal_preprocessor=False):
        file_counter = 0
        directory = self._base_dir + DIR_TF_RECORDS
        for record_type in self._data_base.no_subject.keys():
            for task in self._data_base.get_task_by_type_no_subject(record_type).keys():
                print("Creating tf Records for {} --> {}".format(record_type, task))
                for handler in self._data_base.get_trigger_list_no_subject(record_type, task):
                    self._process_and_save_data(handler, record_type, task, directory, use_signal_preprocessor,
                                                file_counter)
                    file_counter += 1

        print("Finished...")

    def _preprocess_signals(self, data, fs):  # TODO: Write / call signal processor from here...
        return data

    def _save_to_tf_record(self, filename, data, record_type_label, task_label):
        import tensorflow as tf

        def _byte_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        m, n = data.shape
        example = tf.train.Example(features=tf.train.Features(feature={
            DATA: _byte_feature(data.flatten().astype(np.float32).tostring()),  # np can be also serialized
            'm': _int64_feature(m),
            'n': _int64_feature(n),
            RECORD_TYPE_LABEL: _int64_feature(RECORD_TO_NUM[record_type_label]),
            TASK_LABEL: _int64_feature(TASK_TO_NUM[task_label])
        }))

        with tf.python_io.TFRecordWriter(filename) as writer:
            writer.write(example.SerializeToString())


if __name__ == '__main__':
    base_dir = "D:/BCI_stuff/databases/"  # MTA TTK
    # base_dir = 'D:/Csabi/'  # Home
    # base_dir = "D:/Users/Csabi/data"  # ITK
    # base_dir = "/home/csabi/databases/"  # linux

    proc = DBProcessor(base_dir, fast_load=True).use_physionet()
    proc.run()
    proc.check_db_subject()
    # proc.check_db_no_subject()

    # proc.convert_data_to_tfRecords_no_subject()
    # proc.convert_data_to_tfRecords()

    # filename = "/home/csabi/databases/physionet.org/physiobank/database/eegmmidb/S001/S001R03.edf"
    # edf = EDFHandler(filename)
    # print(edf.get_record_number())
    # print(edf.get_subject_number())
    # dat = edf.get_data()
    # print(dat, np.shape(dat))
    # print(edf.get_channels())

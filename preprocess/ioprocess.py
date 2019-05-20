import numpy as np
import mne
from config import *

EPOCH_DB = 'preprocessed_database'


def open_raw_file(filename, preload=True, stim_channel='auto'):
    ext = filename.split('.')[-1]

    switcher = {
        'edf': mne.io.read_raw_edf,
        'eeg': mne.io.read_raw_brainvision,  # todo: check...
    }

    # Get the function from switcher dictionary
    mne_io_read_raw = switcher.get(ext, lambda: "nothing")
    # Execute the function
    return mne_io_read_raw(filename, preload=preload, stim_channel=stim_channel)


def get_filenames_in(path, ext='', recursive=True):
    """
    Searches for files in the given path with specified extension

    :param path: path where to do the search
    :param ext: file extension to search for
    :return: list of filenames
    :raises FileNotFoundError if no files were found
    """
    import glob
    if ext is None:
        ext = ''
    files = glob.glob(path + '/**/*' + ext, recursive=recursive)
    if not files:
        raise FileNotFoundError('Files are not available in path: {}'.format(path))
    return files


def make_dir(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)


# def find_filenames(rec_nums, filenames):
#     import re
#     files = []
#     for i in rec_nums:
#         for fname in filenames:
#             res = re.findall(r'.*R{:02d}.*'.format(i), fname)
#             if res:
#                 files.append(res[0])
#     # files = [ for i in rec_nums for fname in filenames]
#     return files


def get_num_with_predefined_char(filename, char, required_num='required'):
    """
    Searches for number in filename

    :param char: character to search for
    :param required_num: string to print in error message
    :return: number after given char
    :raises FileNotFoundError if none numbers were found
    """
    import re
    num_list = re.findall(r'.*' + char + '\d+', filename)
    if not num_list:
        raise FileNotFoundError(
            "Can not give back {} number: filename '{}' does not contain '{}' character.".format(required_num,
                                                                                                 filename,
                                                                                                 char))
    num_str = num_list[0]
    num_ind = num_str.rfind(char) - len(num_str) + 1
    return int(num_str[num_ind:])


def get_record_number(filename):
    """
    Only works if record number is stored in filename:
        *R<record num>*

    :return: record number from filename
    """
    return get_num_with_predefined_char(filename, 'R', "RECORD")


def get_subject_number(filename):
    """
    Only works if subject number is stored in filename:
        *S<record num>*

    :return: record number from filename
    """
    return get_num_with_predefined_char(filename, 'S', "SUBJECT")


class EEGFileHandler:

    def __init__(self, filename, preload=False, tmin=0, tmax=None, labels=None):
        self._filename = filename
        self._file_handler = None
        self._tmin = tmin  # beggining of raw, In seconds!!! use fs!
        self._tmax = tmax

        if preload:
            self._file_handler = open_raw_file(filename)

        self.labels = labels  # type, subject, task
        if labels is None:
            self.labels = list()

    def _load_file(self):
        if self._file_handler is None:
            raw = open_raw_file(self._filename)
            self._file_handler = raw.crop(self._tmin, self._tmax)

    def create_epochs(self, epoch_dict=None, tmin=0, tmax=4, preload=False):
        self._load_file()
        events = mne.find_events(self._file_handler, shortest_event=0, stim_channel='STI 014', initial_event=True,
                                 consecutive=True)
        epochs = mne.Epochs(self._file_handler, events, event_id=epoch_dict, tmin=tmin, tmax=tmax,
                            proj=True, baseline=None, preload=preload)
        return epochs

    def set_crop_values(self, tmin, tmax):
        self._tmin = tmin
        self._tmax = tmax

    def get_frequency(self):
        self._load_file()
        return self._file_handler.info['sfreq']

    def get_channels(self, remove_trigger=True):
        self._load_file()
        channels = self._file_handler.info['ch_names']
        if remove_trigger:
            channels = channels[:-1]
        return channels

    def get_data(self, remove_trigger=True):
        self._load_file()
        data = self._file_handler.get_data()
        if remove_trigger:
            data = data[:-1, :]
        return data

    def get_mne_object(self):
        self._load_file()
        mne_obj = self._file_handler.copy()
        self.close()
        return mne_obj

    def close(self):
        self._file_handler.close()
        self._file_handler = None


class DataBaseHandler:

    def __init__(self):
        self._db = list()

    def get_filtered_db(self, labels):
        f_res = [eeg_handler for eeg_handler in self._db if all(f_op in eeg_handler.labels for f_op in labels)]
        return f_res

    def append_element(self, eeg_handler):
        self._db.append(eeg_handler)

    def ger_db(self):
        return self._db


class OfflineEpochCreator:
    """
    Preprocessor for edf files. Creates a database, which has all the required information about the eeg files.
    """

    def __init__(self, base_dir, use_drop_subject_list=True):
        self._base_dir = base_dir
        self._data_path = None
        self._data_duration = 4  # seconds
        self._db_type = SourceDB()  # Physionet / TTK
        self._db_handler = DataBaseHandler()

        # self._db_ext = None
        # self._trigger_task_list = None

        self._drop_subject = set()
        if use_drop_subject_list:
            self._drop_subject.add(-1)

        if not base_dir[-1] == '/':
            self._base_dir = base_dir + '/'

    def _use_db(self, db_type):
        """
        Loads a specified database.

        :param path: where the database is available
        """
        self._data_path = self._base_dir + db_type.DIR
        self._db_type = db_type
        self._drop_subject = db_type.DROP_SUBJECTS

    def use_physionet(self):
        self._use_db(Physionet)
        return self

    @property
    def _db_ext(self):
        return self._db_type.DB_EXT

    # @property
    # def _trigger_task_list(self):
    #     return self._db_type.TRIGGER_TASK_LIST

    def _conv_type(self, record_num):
        return self._db_type.TRIGGER_TYPE_CONVERTER.get(record_num)

    def _conv_task(self, record_num, task_ID):
        return self._db_type.TRIGGER_TASK_CONVERTER.get(record_num).get(task_ID)

    def run(self):
        self._create_annotated_db()

    """
    Database functions
    """

    def _get_filenames(self):
        return get_filenames_in(self._data_path, self._db_ext)

    def convert_type(self, record_number):
        return self._db_type.TRIGGER_TYPE_CONVERTER.get(record_number)

    def convert_task(self, record_number):
        return self._db_type.TRIGGER_TASK_CONVERTER.get(record_number)

    def _create_annotated_db(self):
        filenames = self._get_filenames()
        for file in filenames:
            rec_num = get_record_number(file)
            subj_num = get_subject_number(file)

            if subj_num in self._drop_subject:
                continue

            eeg = EEGFileHandler(file, preload=True)
            print("valami", self.convert_task(rec_num))
            epochs = eeg.create_epochs(self.convert_task(rec_num))
            print(epochs)
            break


if __name__ == '__main__':
    # base_dir = "D:/BCI_stuff/databases/"  # MTA TTK
    # base_dir = 'D:/Csabi/'  # Home
    # base_dir = "D:/Users/Csabi/data"  # ITK
    base_dir = "/home/csabi/databases/"  # linux

    proc = OfflineEpochCreator(base_dir).use_physionet()
    proc.run()

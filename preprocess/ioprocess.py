import numpy as np
import mne
from config import *

EPOCH_DB = 'preprocessed_database'


def open_raw_file(filename, preload=True):
    """
    Wrapper function to open either edf or brainvision files.

    :param filename: filename of raw file
    :param preload: load data to memory
    :return: raw mne file
    """
    ext = filename.split('.')[-1]

    if ext == 'edf':
        raw = mne.io.read_raw_edf(filename, preload=preload)
    elif ext == 'vhdr':
        raw = mne.io.read_raw_brainvision(filename, preload=preload)
    else:
        raw = None

    return raw


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


def filter_filenames(files, subject, runs):
    """
    Filter file names for one subject and for given record numbers
    Only for Physionet data

    :param files: list of str
    :param subject: int
    :param runs: list of int
    :return: list of str
    """
    rec = list()
    subj = [f for f in files if subject == get_subject_number(f)]

    for s in subj:
        for i in runs:
            if i == get_record_number(s):
                rec.append(s)

    return rec


def generate_filenames(file_path, subject, runs):
    rec = list()
    for i in runs:
        f = file_path
        f = f.replace('{subj}', '{:03d}'.format(subject))
        f = f.replace('{rec}', '{:02d}'.format(i))
        rec.append(f)

    return rec


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
        import warnings
        warnings.warn("Can not give back {} number: filename '{}' does not contain '{}' character.".format(required_num,
                                                                                                           filename,
                                                                                                           char))
        return None

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


class SubjectKFold(object):
    """
    Class to split subject db to train and test
    """

    def __init__(self, k_fold_num=None, shuffle=True, random_state=None):
        self._k_fold_num = k_fold_num
        self._shuffle = shuffle
        self._random_state = random_state

    def split(self, subject_db):
        ind = np.arange(len(subject_db)) + 1

        if self._shuffle:
            np.random.seed(self._random_state)
            np.random.shuffle(ind)

        for i in ind:
            yield subject_db.get_split(i)


"""
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
"""


class OfflineEpochCreator:
    """
    Preprocessor for edf files. Creates a database, which has all the required information about the eeg files.
    TODO: check what can be removed!!!
    """

    def __init__(self, base_dir, data_duration=3, use_drop_subject_list=True):
        self._base_dir = base_dir
        self._data_path = None
        self._data_duration = data_duration  # seconds
        self._db_type = None  # Physionet / TTK

        self._drop_subject = None
        if use_drop_subject_list:
            self._drop_subject = set()

        if not base_dir[-1] == '/':
            self._base_dir = base_dir + '/'

    def _use_db(self, db_type):
        """
        Loads a specified database.

        :param path: where the database is available
        """
        self._data_path = self._base_dir + db_type.DIR
        self._db_type = db_type

        if self._drop_subject is not None:
            self._drop_subject = set(db_type.DROP_SUBJECTS)
        else:
            self._drop_subject = set()

    def use_physionet(self):
        self._use_db(Physionet)
        return self

    def use_pilot(self):
        self._use_db(PilotDB)
        return self

    @property
    def _db_ext(self):
        return self._db_type.DB_EXT

    # def _conv_type(self, record_num):
    #     return self._db_type.TRIGGER_TYPE_CONVERTER.get(record_num)
    #
    # def _conv_task(self, record_num, task_ID):
    #     return self._db_type.TRIGGER_TASK_CONVERTER.get(record_num).get(task_ID)

    def run(self):
        self._create_db()

    """
    Database functions
    """

    def _get_filenames(self):
        return get_filenames_in(self._data_path, self._db_ext)

    def convert_type(self, record_number):
        return self._db_type.TRIGGER_CONV_REC_TO_TYPE.get(record_number)

    def convert_task(self, record_number=None):
        if record_number is None:
            return self._db_type.TRIGGER_TASK_CONVERTER
        return self._db_type.TRIGGER_CONV_REC_TO_TASK.get(record_number)

    def convert_rask_to_recs(self, task):
        return self._db_type.TASK_TO_REC.get(task)

    def get_trigger_event_id(self):
        return self._db_type.TRIGGER_EVENT_ID

    def _create_physionet_db(self):

        keys = self._db_type.TASK_TO_REC.keys()

        for s in range(self._db_type.SUBJECT_NUM):
            subj = s + 1

            if subj in self._drop_subject:
                continue

            for task in keys:
                filenames = generate_filenames(self._data_path + self._db_type.FILE_PATH, subj,
                                               self.convert_rask_to_recs(task))
                raws = [open_raw_file(file, preload=False) for file in filenames]
                raw = raws.pop(0)
                for r in raws:
                    raw.append(r)
                del raws

                # todo: make filtering here...

                rec_num = get_record_number(filenames[0])
                task_dict = self.convert_task(rec_num)

                events, _ = mne.events_from_annotations(raw)
                epochs = mne.Epochs(raw, events, event_id=task_dict, tmin=0, tmax=3, preload=False)
                # todo: make windowing!

    def _create_db(self):
        filenames = self._get_filenames()
        # layout = mne.channels.read_layout('EEG1005')

        if self._db_type is Physionet:
            self._create_physionet_db()
        else:
            raise NotImplementedError('Cannot create subject database for {}'.format(self._db_type))

    # def _create_annotated_db(self):
    #     filenames = self._get_filenames()
    #     for file in filenames:
    #         rec_num = get_record_number(file)
    #         subj_num = get_subject_number(file)
    #
    #         if subj_num in self._drop_subject:
    #             continue
    #
    #         eeg = EEGFileHandler(file, preload=True)
    #         print("valami", self.convert_task(rec_num))
    #         epochs = eeg.create_epochs(self.convert_task(rec_num))
    #         print(epochs)
    #         break


if __name__ == '__main__':
    base_dir = "D:/BCI_stuff/databases/"  # MTA TTK
    # base_dir = 'D:/Csabi/'  # Home
    # base_dir = "D:/Users/Csabi/data"  # ITK
    # base_dir = "/home/csabi/databases/"  # linux

    proc = OfflineEpochCreator(base_dir).use_physionet()
    proc.run()

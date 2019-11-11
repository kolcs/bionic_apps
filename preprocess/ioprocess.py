import numpy as np
import pickle
import mne
import time
from sklearn.model_selection import KFold

from config import *
from preprocess.feature_extraction import calculate_spatial_data, calculate_fft_power, calculate_fft_range

EPOCH_DB = 'preprocessed_database'

# features
SPATIAL = 'spatial'
AVG_COLUMN = 'avg_column'
COLUMN = 'column'
FFT_POWER = 'fft_power'
FFT_RANGE = 'fft_range'

# config options
CONFIG_FILE = 'bci_system.cfg'
BASE_DIR = 'base_dir'


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
        raise NotImplementedError('{} file reading is not implemented.'.format(ext))

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
    """
    Creates dir if it does not exists on given path

    :param path: str
        path to new dir
    """
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


def _generate_filenames_for_subject(file_path, subject, subject_format_str, runs, run_format_str):
    """Filename generator for one subject

    Generating filenames from a given string with {subj} and {rec} which will be replaced.

    Parameters
    ----------
    file_path : str
        special string with {subj} and {rec} substrings
    subject : int
        subject number
    subject_format_str : str
        The string which will be formatted.
    runs : int or list of int
        list of sessions
    run_format_str : str
        The string which will be formatted.

    Returns
    -------
    rec : list of str
        filenames for a subject with given runs

    """
    if type(runs) is not list:
        runs = [runs]

    rec = list()
    for i in runs:
        f = file_path
        f = f.replace('{subj}', subject_format_str.format(subject))
        f = f.replace('{rec}', run_format_str.format(i))
        rec.append(f)

    return rec


def generate_physionet_filenames(file_path, subject, runs):
    """Filename generator for pyhsionet db

    Generating filenames from a given string with {subj} and {rec} which are will be replaced.

    Parameters
    ----------
    file_path : str
        special string with {subj} and {rec} substrings
    subject : int
        subject number
    runs : int or list of int
        list of sessions

    Returns
    -------
    rec : list of str
        filenames for a subject with given runs

    """
    return _generate_filenames_for_subject(file_path, subject, '{:03d}', runs, '{:02d}')


def generate_pilot_filenames(file_path, subject, runs=1):
    """Filename generator for pilot db

        Generating filenames from a given string with {subj} and {rec} which are will be replaced.

        Parameters
        ----------
        file_path : str
            special string with {subj} and {rec} substrings
        subject : int
            subject number
        runs : int or list of int
            list of sessions

        Returns
        -------
        rec : list of str
            filenames for a subject with given runs

        """
    return _generate_filenames_for_subject(file_path, subject, '{}', runs, '{:02d}')


def generate_ttk_filenames(file_path, subject, runs=1):
    """Filename generator for pilot db

        Generating filenames from a given string with {subj} and {rec} which are will be replaced.

        Parameters
        ----------
        file_path : str
            special string with {subj} and {rec} substrings
        subject : int
            subject number
        runs : int or list of int
            list of sessions

        Returns
        -------
        rec : list of str
            filenames for a subject with given runs

        """
    return _generate_filenames_for_subject(file_path, subject, '{:02d}', runs, '{:02d}')


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


def load_pickle_data(filename):
    import os
    if not os.path.exists(filename):
        return None

    with open(filename, 'rb') as fin:
        data = pickle.load(fin)
    return data


def save_pickle_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def init_base_config(path='./'):
    from os.path import realpath, dirname, join
    file_dir = dirname(realpath('__file__'))
    file = join(file_dir, path + CONFIG_FILE)
    cfg_dict = load_pickle_data(file)
    if cfg_dict is None:
        from gui_handler import select_base_dir
        base_directory = select_base_dir()
        cfg_dict = {BASE_DIR: base_directory}
        save_pickle_data(cfg_dict, file)
    else:
        base_directory = cfg_dict[BASE_DIR]
    return base_directory


def get_epochs_from_files(filenames, task_dict, epoch_tmin=-0.2, epoch_tmax=0.5):
    """Generate epochs from files.

    Parameters
    ----------
    filenames : str, list of str
        List of file names from where epochs will be generated.
    task_dict : dict
        Used for creating mne.Epochs.
    epoch_tmin : float
        Start time before event. If nothing is provided, defaults to -0.2
    epoch_tmax : float
        End time after event. If nothing is provided, defaults to 0.5

    Returns
    -------
    mne.Epochs
        Created epochs from files.

    """
    if type(filenames) is str:
        filenames = [filenames]
    raws = [open_raw_file(file, preload=False) for file in filenames]
    raw = raws.pop(0)

    fs = raw.info['sfreq']

    for r in raws:
        raw.append(r)
    del raws
    raw.rename_channels(lambda x: x.strip('.'))

    # todo: make prefiltering here...
    # raw.load_data()
    # iir_params = dict(order=5, ftype='butter', output='sos')
    # raw.filter(l_freq=.5, h_freq=35, method='iir', iir_params=iir_params)

    events, _ = mne.events_from_annotations(raw)
    baseline = None  # tuple([None, self._epoch_tmin + 0.1])  # if self._epoch_tmin > 0 else (None, 0)
    epochs = mne.Epochs(raw, events, baseline=baseline, event_id=task_dict, tmin=epoch_tmin,
                        tmax=epoch_tmax, preload=False)
    return epochs, fs


class SubjectKFold(object):
    """
    Class to split subject databse to train and test
    """

    def __init__(self, k_fold_num=None, shuffle_subjects=True, shuffle_data=True, random_state=None):
        self._k_fold_num = k_fold_num
        self._shuffle_subj = shuffle_subjects
        self._shuffle_data = shuffle_data
        self._random_state = random_state

    def split(self, subject_db):
        subjects = subject_db.get_subjects()

        if self._shuffle_subj:
            np.random.seed(self._random_state)
            np.random.shuffle(subjects)

        if self._k_fold_num is not None:
            self._k_fold_num = max(1, self._k_fold_num)
            n = min(self._k_fold_num, len(subjects))
            subjects = subjects[:n]

        for s in subjects:
            yield subject_db.get_split(s, shuffle=self._shuffle_data, random_seed=self._random_state)

    def split_subject_data(self, subject_db, subject, k_fold_num=None, shuffle=True, random_seed=None):
        if k_fold_num is not None:
            self._k_fold_num = k_fold_num
        if self._k_fold_num is None:
            self._k_fold_num = 10

        data, labels = subject_db.get_subject_data(subject, shuffle=shuffle, random_seed=random_seed)
        label_inds = {label: list() for label in set(labels)}
        for i, label in enumerate(labels):
            label_inds[label].append(i)

        prev_inds = list(label_inds.values())[0]
        for label, inds in label_inds.items():
            assert len(inds) == len(prev_inds), 'The number of label instances are not equal.'
            prev_inds = inds

        kf = KFold(n_splits=self._k_fold_num)
        for train_split_ind, test_split_ind in kf.split(prev_inds):
            train_ind = list()
            test_ind = list()
            for label, inds in label_inds.items():
                train_ind.extend(np.array(inds)[train_split_ind])
                test_ind.extend(np.array(inds)[test_split_ind])

            if shuffle:
                np.random.seed(random_seed)
                np.random.shuffle(train_ind)
                np.random.seed(random_seed)
                np.random.shuffle(test_ind)

            train_x = [data[ind] for ind in train_ind]
            train_y = [labels[ind] for ind in train_ind]
            test_x = [data[ind] for ind in test_ind]
            test_y = [labels[ind] for ind in test_ind]

            yield train_x, train_y, test_x, test_y


class OfflineDataPreprocessor:
    """
    Preprocessor for edf files. Creates a database, which has all the required information about the eeg files.
    TODO: check what can be removed!!!
    """

    def __init__(self, base_dir, epoch_tmin=0, epoch_tmax=3, use_drop_subject_list=True, window_length=0.5,
                 window_step=0.25, fast_load=True):

        self._base_dir = base_dir
        self._epoch_tmin = epoch_tmin
        self._epoch_tmax = epoch_tmax  # seconds
        self._window_length = window_length  # seconds
        self._window_step = window_step  # seconds
        self._fast_load = fast_load

        self._fs = int()
        self._feature = str()
        self._fft_low = int()
        self._fft_high = int()
        self._fft_step = int()
        self._fft_width = int()
        self._data_set = dict()

        self._interp = None
        self._data_path = None
        self._db_type = None  # Physionet / TTK

        self.proc_db_path = ''
        self._proc_db_filenames = list()
        self._proc_db_source = ''

        self._drop_subject = set() if use_drop_subject_list else None

        if not base_dir[-1] == '/':
            self._base_dir = base_dir + '/'

    def _use_db(self, db_type):
        """
        Loads a specified database.
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

    def use_pilot_par_b(self):
        self._use_db(PilotDB_ParadigmB)
        return self

    def use_ttk_db(self):
        self._use_db(TTK_DB)
        return self

    def use_game_data(self):
        self._use_db(GameDB)

    @property
    def _db_ext(self):
        return self._db_type.DB_EXT

    def run(self, feature='avg_column', fft_low=7, fft_high=13, fft_step=2, fft_width=2):
        """Runs the Database preprocessor with the given features.

        Parameters
        ----------
        feature: 'spatial' | 'avg_column' | 'column' | None (default 'spatial')
            The feature which will be created.

        """
        self._feature = feature
        self._fft_low = fft_low
        self._fft_high = fft_high
        self._fft_step = fft_step
        self._fft_width = fft_width
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

    def get_subjects(self):
        return list(self._data_set.keys())

    def get_command_converter(self):
        attr = 'COMMAND_CONV'
        assert hasattr(self._db_type, attr), '{} has no {} attribute'.format(self._db_type, attr)
        return self._db_type.COMMAND_CONV

    def _load_data_from_source(self, source_files):
        """Loads preprocessed feature data form source files.

        Parameters
        ----------
        source_files : list of str
            List of source file names without path.

        Returns
        -------
        dict
            Preprocessed data dictionary, where the keys are the subjects in the database and
            the values are the list of preprocessed feature data.

        """
        data = dict()
        for filename in source_files:
            d = load_pickle_data(self.proc_db_path + filename)
            data.update(d)

        return data

    def _save_preprocessed_subject_data(self, subject_data, subj):
        """Saving preprocessed feature data for a given subject.

        Parameters
        ----------
        subject_data : list
            Featured and preprocessed data.
        subj : int
            Subject number
        """
        self._data_set[subj] = subject_data

        db_file = 'subject{}.data'.format(subj)
        save_pickle_data({subj: subject_data}, self.proc_db_path + db_file)
        self._proc_db_filenames.append(db_file)
        # save_pickle_data(self._proc_db_filenames, self._proc_db_source)

    def _create_physionet_db(self):
        """Physionet feature db creator function"""

        keys = self._db_type.TASK_TO_REC.keys()

        for s in range(self._db_type.SUBJECT_NUM):
            subj = s + 1

            if subj in self._drop_subject:
                print('Dropping subject {}'.format(subj))
                continue

            subject_data = list()

            for task in keys:
                recs = self.convert_rask_to_recs(task)
                task_dict = self.convert_task(recs[0])
                filenames = generate_physionet_filenames(self._data_path + self._db_type.FILE_PATH, subj,
                                                         recs)
                epochs, self._fs = get_epochs_from_files(filenames, task_dict, self._epoch_tmin, self._epoch_tmax)
                win_epochs = self._get_windowed_features(epochs, task)

                subject_data.extend(win_epochs)

            self._save_preprocessed_subject_data(subject_data, subj)
        save_pickle_data(self._proc_db_filenames, self._proc_db_source)

    def _create_ttk_db(self):
        """Pilot feature db creator function"""
        task_dict = self.convert_task()

        for s in range(self._db_type.SUBJECT_NUM):
            subj = s + 1
            if subj in self._drop_subject:
                print('Dropping subject {}'.format(subj))
                continue

            subject_data = list()
            if self._db_type is TTK_DB:
                filenames = generate_ttk_filenames(self._data_path + self._db_type.FILE_PATH, subj)
            else:
                filenames = generate_pilot_filenames(self._data_path + self._db_type.FILE_PATH, subj)
            epochs, self._fs = get_epochs_from_files(filenames, task_dict, self._epoch_tmin, self._epoch_tmax)

            for task in task_dict.keys():
                win_epochs = self._get_windowed_features(epochs, task)
                subject_data.extend(win_epochs)

            self._save_preprocessed_subject_data(subject_data, subj)
        save_pickle_data(self._proc_db_filenames, self._proc_db_source)

    def _create_game_db(self):
        from gui_handler import select_eeg_file_in_explorer
        filename = select_eeg_file_in_explorer(self._base_dir)
        assert filename is not None, 'No source files were selected. Cannot play BCI game.'

        task_dict = self.convert_task()
        subject_data = list()
        epochs, self._fs = get_epochs_from_files(filename, task_dict, self._epoch_tmin, self._epoch_tmax)
        for task in task_dict.keys():
            win_epochs = self._get_windowed_features(epochs, task)
            subject_data.extend(win_epochs)
        self._data_set[0] = subject_data

    def _get_windowed_features(self, epochs, task):
        """Feature creation from windowed data.

        self._feature : {'spatial', 'avg_column', 'column'}, optional
            The feature which will be created.

        Parameters
        ----------
        epochs : mne.Epochs
            Mne epochs from which the feature will be created.
        task : str
            Selected from epochs.

        Returns
        -------
        list
            Windowed feature data.

        """
        epochs = epochs[task]

        win_epochs = list()
        window_length = self._window_length - 1 / self._fs  # win length correction
        win_num = int((self._epoch_tmax - self._epoch_tmin - window_length) / self._window_step) \
            if self._window_step > 0 else 1

        for i in range(win_num):
            ep = epochs.copy().load_data()
            ep.crop(ep.tmin + i * self._window_step, ep.tmin + window_length + i * self._window_step)

            if self._feature == SPATIAL:
                data, self._interp = calculate_spatial_data(self._interp, ep)
            elif self._feature == AVG_COLUMN:
                data = ep.get_data()
                data = np.average(data, axis=-1)  # average window
            elif self._feature == COLUMN:
                data = ep.get_data()
                (epoch, channel, time) = np.shape(data)
                data = np.reshape(data, (epoch, channel * time))
            elif self._feature == FFT_POWER:
                data = calculate_fft_power(ep.get_data(), self._fs, self._fft_low, self._fft_high)
            elif self._feature == FFT_RANGE:
                data = calculate_fft_range(ep.get_data(), self._fs, self._fft_low, self._fft_high, self._fft_step,
                                           self._fft_width)
            else:
                raise NotImplementedError('{} feature creation is not implemented'.format(self._feature))

            data = [(d, task) for d in data]
            win_epochs.extend(data)

        return win_epochs

    def init_processed_db_path(self, feature=None):
        if feature is not None:
            self._feature = feature
        self.proc_db_path = "{}{}{}{}/".format(self._base_dir, DIR_FEATURE_DB, self._db_type.DIR, self._feature)

    def _create_db(self):
        """Base db creator function."""

        assert self._db_type is not None, \
            'Define a database with .use_<db_name>() function before creating the database!'
        self._data_set = dict()
        tic = time.time()
        self.init_processed_db_path()
        self._proc_db_source = self.proc_db_path + self._feature + '.db'
        make_dir(self.proc_db_path)

        def print_creation_message():
            print('{} file is not found. Creating database.'.format(file))

        file = self._proc_db_source
        data_source = load_pickle_data(file)

        if data_source is not None and self._fast_load:
            self._data_set = self._load_data_from_source(data_source)
            for subj in self._drop_subject:
                self._data_set.pop(subj, 'No error!')

        elif self._db_type is Physionet:
            print_creation_message()
            self._create_physionet_db()

        elif self._db_type in [PilotDB, PilotDB_ParadigmB, TTK_DB]:
            print_creation_message()
            self._create_ttk_db()

        elif self._db_type is GameDB:
            self._create_game_db()

        else:
            raise NotImplementedError('Cannot create subject database for {}'.format(self._db_type))

        print('Database initialization took {} seconds.'.format(int(time.time() - tic)))

    def get_subject_data(self, subject, reduce_rest=True, shuffle=True, random_seed=None):
        """Returns data for one subject.

        Parameters
        ----------
        subject : int
            Number of the required subject.
        reduce_rest : bool, optional
            To reduce rest data points.
        """
        assert subject in list(self._data_set.keys()), \
            'Subject{} is not in preprocessed database'.format(subject)
        data, label = zip(*self._data_set[subject])
        if reduce_rest:
            data, label = self._reduce_rest_label(data, label)
        if shuffle:
            ind = np.arange(len(label))
            np.random.seed(random_seed)
            np.random.shuffle(ind)
            data = [data[i] for i in ind]
            label = [label[i] for i in ind]

        return list(data), list(label)

    def get_split(self, test_subject, shuffle=True, random_seed=None, reduce_rest=True):
        """Splits the whole database to train and test sets.

        This is a helper function for :class:`SubjectKFold` to make a split from the whole database.
        This function creates a train and test set. The test set is the data of the given subject
        and the train set is the rest of the database.

        Parameters
        ----------
        test_subject : int
            Subject to put into the train set.
        shuffle : bool, optional
            Shuffle the train set if True
        random_seed : int, optional
            Random seed for shuffle.
        reduce_rest : bool, optional
            Make rest number equal to the max of other labels.

        Returns
        -------

        """
        train_subjects = list(self._data_set.keys())
        # if type(test_subject) is int:
        #     test_subject = [test_subject]
        train_subjects.remove(test_subject)
        # train_subjects = [subj for subj in train_subjects if subj not in test_subject]

        train = list()
        test = self._data_set.get(test_subject)

        # for s in test_subject:
        #     test.extend(self._data_set.get(s))

        for s in train_subjects:
            train.extend(self._data_set.get(s))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(train)
            np.random.seed(random_seed)
            np.random.shuffle(test)

        train_x, train_y = zip(*train)
        test_x, test_y = zip(*test)

        if reduce_rest:
            train_x, train_y = self._reduce_rest_label(train_x, train_y)
            test_x, test_y = self._reduce_rest_label(test_x, test_y)

        return list(train_x), list(train_y), list(test_x), list(test_y), test_subject

    @staticmethod
    def _reduce_rest_label(data, labels):
        from config import REST
        label_count = {lab: labels.count(lab) for lab in set(labels)}
        rest_count = label_count[REST]
        del label_count[REST]
        max_count = max(label_count.values())
        rest_ind = [i for i, lab in enumerate(labels) if lab == REST]
        del_num = rest_count - max_count
        labels = np.delete(labels, rest_ind[:del_num])
        data = np.delete(data, rest_ind[:del_num], axis=0)
        return data, labels


if __name__ == '__main__':
    base_dir = init_base_config('../')

    preprocessor = OfflineDataPreprocessor(base_dir, fast_load=True).use_pilot()
    preprocessor.run(feature=FFT_POWER)

    # this is how SubjectKFold works:
    subj_k_fold = SubjectKFold(10)

    for train_x, train_y, test_x, test_y in subj_k_fold.split_subject_data(preprocessor, 1):
        print(np.array(train_x).shape, len(train_y), len(test_x), len(test_y))
        break

    for train_x, train_y, test_x, test_y, subject in subj_k_fold.split(preprocessor):
        print(train_x[0], train_y[0], test_x[0], test_y[0], subject)
        break

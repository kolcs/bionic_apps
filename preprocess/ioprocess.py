from enum import Enum
from json import dump as json_dump, load as json_load
from pathlib import Path
from pickle import dump as pkl_dump, load as pkl_load
from time import time

import mne
import numpy as np
from sklearn.model_selection import KFold

from config import Physionet, PilotDB_ParadigmA, PilotDB_ParadigmB, TTK_DB, GameDB, Game_ParadigmC, Game_ParadigmD, \
    DIR_FEATURE_DB, REST, CALM, ACTIVE
from gui_handler import select_file_in_explorer
from preprocess.feature_extraction import FeatureType, FeatureExtractor

EPOCH_DB = 'preprocessed_database'

# config options
CONFIG_FILE = 'bci_system.cfg'
BASE_DIR = 'base_dir'
SUBJECTS = 'subjects'
DATA_SOURCE = 'data_source'
INFO = 'epoch_info'


# db selection options
class Databases(Enum):
    PHYSIONET = 'physionet'
    PILOT_PAR_A = 'pilot_par_a'
    PILOT_PAR_B = 'pilot_par_b'
    TTK = 'ttk'
    GAME = 'game'
    GAME_PAR_C = 'game_par_c'
    GAME_PAR_D = 'game_par_d'


def open_raw_with_gui():
    return open_raw_file(select_file_in_explorer(init_base_config()))


def open_raw_file(filename, preload=True, **mne_kwargs):
    """Wrapper function to open either edf or brainvision files.

    Parameters
    ----------
    filename : str
        Absolute path and filename of raw file.
    preload : bool
        Load data to memory.
    mne_kwargs
        Arbitrary keywords for mne file opener functions.

    Returns
    -------
    mne.Raw
        Raw mne file.
    """
    ext = Path(filename).suffix

    if ext == '.edf':
        raw = mne.io.read_raw_edf(filename, preload=preload, **mne_kwargs)
    elif ext == '.vhdr':
        raw = mne.io.read_raw_brainvision(filename, preload=preload, **mne_kwargs)
    else:
        raise NotImplementedError('{} file reading is not implemented.'.format(ext))

    return raw


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


def load_pickle_data(filename):
    if not Path(filename).exists():
        return None

    with open(filename, 'rb') as fin:
        data = pkl_load(fin)
    return data


def save_pickle_data(filename, data):
    with open(filename, 'wb') as f:
        pkl_dump(data, f)


def load_from_json(file):
    with open(file) as json_file:
        data_dict = json_load(json_file)
    return data_dict


def save_to_json(file, data_dict):
    with open(file, 'w') as outfile:
        json_dump(data_dict, outfile, indent='\t')


def init_base_config(path='.'):
    """Loads base directory path from pickle data. It it does not exist it creates it.

    Parameters
    ----------
    path : str
        Relative path to search for config file.

    Returns
    -------
    str
        Base directory path.
    """
    file_dir = Path('.').resolve()
    file = file_dir.joinpath(path, CONFIG_FILE)
    cfg_dict = load_pickle_data(file)
    if cfg_dict is None:
        from gui_handler import select_base_dir
        base_directory = select_base_dir()
        cfg_dict = {BASE_DIR: base_directory}
        save_pickle_data(file, cfg_dict)
    else:
        base_directory = cfg_dict[BASE_DIR]
    return base_directory


def pollute_data(dataset, dev=0.000005):  # todo: integrate to the system and test it!
    """Use it for data augmentation
    This function adds gautian noise to the eeg data.
    """
    noise = np.random.normal(0, dev, dataset.shape)
    return np.add(dataset, noise)


def _cut_real_movemet_data(raw):
    from pandas import DataFrame, concat
    raw.load_data()
    events, _ = mne.events_from_annotations(raw, event_id='auto')
    df = DataFrame(events, columns=['time', 'none', 'id'])
    resp = df[df.id == 1001]
    start = df[df.id == 16]
    ev = concat((resp, start))
    ev = ev.to_numpy()
    if np.size(ev, 0) > 10:
        t = ev[3, 0] / raw.info['sfreq']
        raw.crop(tmin=t)
    return raw


def get_epochs_from_raw(raw, task_dict, epoch_tmin=-0.2, epoch_tmax=0.5, baseline=None, event_id='auto', preload=True):
    """Generate epochs from files.

    Parameters
    ----------
    raw : mne.Raw
        Raw eeg file.
    task_dict : dict
        Used for creating mne.Epochs.
    epoch_tmin : float
        Start time before event. If nothing is provided, defaults to -0.2
    epoch_tmax : float
        End time after event. If nothing is provided, defaults to 0.5
    baseline : None or (float, float) or (None, float) or (float, None)
        The time interval to apply baseline correction. If None do not apply
        it. If baseline is (a, b) the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used and if b is None then b
        is set to the end of the interval. If baseline is equal to (None, None)
        all the time interval is used. Correction is applied by computing mean
        of the baseline period and subtracting it from the data. The baseline
        (a, b) includes both endpoints, i.e. all timepoints t such that
        a <= t <= b.
    event_id : int or list of int or dict or None
        The id of the event to consider. If dict, the keys can later be used
        to access associated events. Example: dict(auditory=1, visual=3).
        If int, a dict will be created with the id as string. If a list, all
        events with the IDs specified in the list are used. If None, all events
        will be used with and a dict is created with string integer names
        corresponding to the event id integers.
    preload : bool
        Load all epochs from disk when creating the object or wait before
        accessing each epoch (more memory efficient but can be slower).

    Returns
    -------
    mne.Epochs
        Created epochs from files. Data is not loaded!
    """
    events, _ = mne.events_from_annotations(raw, event_id)
    # baseline = tuple([None, epoch_tmin + 0.1])  # if self._epoch_tmin > 0 else (None, 0)
    epochs = mne.Epochs(raw, events, baseline=baseline, event_id=task_dict, tmin=epoch_tmin,
                        tmax=epoch_tmax, preload=preload, on_missing='warning')
    return epochs


def get_epochs_from_files(filenames, task_dict, epoch_tmin=-0.2, epoch_tmax=0.5, baseline=None, event_id='auto',
                          preload=False, prefilter_signal=False, f_type='butter', f_order=5, l_freq=1, h_freq=None,
                          cut_real_movement_tasks=False):
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
    baseline : None or (float, float) or (None, float) or (float, None)
        The time interval to apply baseline correction. If None do not apply
        it. If baseline is (a, b) the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used and if b is None then b
        is set to the end of the interval. If baseline is equal to (None, None)
        all the time interval is used. Correction is applied by computing mean
        of the baseline period and subtracting it from the data. The baseline
        (a, b) includes both endpoints, i.e. all timepoints t such that
        a <= t <= b.
    event_id : int or list of int or dict or None
        The id of the event to consider. If dict, the keys can later be used
        to access associated events. Example: dict(auditory=1, visual=3).
        If int, a dict will be created with the id as string. If a list, all
        events with the IDs specified in the list are used. If None, all events
        will be used with and a dict is created with string integer names
        corresponding to the event id integers.
    preload : bool
        Load all epochs from disk when creating the object or wait before
        accessing each epoch (more memory efficient but can be slower).
    prefilter_signal : bool
        Make signal filtering before preprocess.

    Returns
    -------
    mne.Epochs
        Created epochs from files. Data is not loaded!

    """
    if type(filenames) is str:
        filenames = [filenames]
    raws = [open_raw_file(file, preload=False) for file in filenames]
    raw = raws.pop(0)

    for r in raws:
        raw.append(r)
    del raws
    raw.rename_channels(lambda x: x.strip('.').capitalize())

    if cut_real_movement_tasks:
        raw = _cut_real_movemet_data(raw)

    if prefilter_signal:
        raw.load_data()
        iir_params = dict(order=f_order, ftype=f_type, output='sos')
        raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir', iir_params=iir_params)

    epochs = get_epochs_from_raw(raw, task_dict, epoch_tmin, epoch_tmax, baseline, event_id, preload=preload)

    return epochs


def _same_number_of_labels(task_dict):
    lens = [len(ep_dict) for ep_dict in task_dict.values()]
    first = lens[0]
    for i in range(1, len(lens)):
        if first != lens[i]:
            return False
    return True


def _reduce(task_dict):
    tasks, epoch_dicts = zip(*task_dict.items())
    data_lens = [len(ep_dict) for ep_dict in epoch_dicts]
    max_task = tasks[np.argmax(data_lens)]
    min_len = data_lens[np.argmin(data_lens)]

    ind_list = np.arange(max(data_lens))
    np.random.shuffle(ind_list)
    task_dict[max_task] = {i: task_dict[max_task][ind] for i, ind in enumerate(ind_list[:min_len])}


def _check_data_equality(data_dict):
    lens = [len(d_list) for d_list in data_dict.values()]
    first_len = lens[0]
    for ln in lens:
        assert ln == first_len, 'Number of data are not equal in each task.'


def _generate_window_list_from_epoch_list(epoch_list):
    """Creates a list of windows from list of list of windows"""
    win_list = list()
    for d in epoch_list:
        win_list.extend(d)
    return win_list


def get_db_name_by_filename(filename):
    if Game_ParadigmC.DIR in filename:
        db_name = Databases.GAME_PAR_C
    elif Game_ParadigmD.DIR in filename:
        db_name = Databases.GAME_PAR_D
    elif PilotDB_ParadigmA.DIR in filename:
        db_name = Databases.PILOT_PAR_A
    elif PilotDB_ParadigmB.DIR in filename:
        db_name = Databases.PILOT_PAR_B
    elif Physionet.DIR in filename:
        db_name = Databases.PHYSIONET
    else:
        raise ValueError('No database defined with path {}'.format(filename))
    return db_name


"""
Feature generation methods for train and test
"""


def create_binary_label(label):
    if label is not REST:
        label = CALM if CALM in label else ACTIVE
    return label


class SubjectKFold(object):

    def __init__(self, source_db, k_fold_num=None, shuffle_subjects=True, shuffle_data=True, random_state=None,
                 equalize_labesl=True,  # batch_size=None
                 ):
        """Class to split subject database to train and test set, in an N-fold cross validation manner.

        Parameters
        ----------
        source_db : OfflineDataPreprocessor
            Source database with preprocessed filenames.
        k_fold_num : int, optional
        shuffle_subjects : bool
        shuffle_data : bool
        random_state : int
        """
        self._source_db = source_db
        self._k_fold_num = k_fold_num
        self._shuffle_subj = shuffle_subjects
        self._shuffle_data = shuffle_data
        np.random.seed(random_state)
        self._equalize_labels = equalize_labesl
        # self._batch_size = batch_size

    def get_individual_class_labels(self, make_binary_label=False):
        labels = self._source_db.get_labels()
        if make_binary_label:
            labels = list(set([create_binary_label(label) for label in labels]))
        return labels

    def split(self, subject=None):
        """Split database to train and test sets in k-fold manner.

        Parameters
        ----------
        subject : int or None
            If subject is not defined the whole database will be split according
            to generate cross subject data. Otherwise only the subject data
            will be split.

        Returns
        -------

        """
        if subject is None:
            raise NotImplementedError
            # if self._batch_size is None:
            #     return self.pregen_split_cross_subject_data()
            # return self.split_db()

        else:
            if self._k_fold_num is None:
                self._k_fold_num = 5

            db_dict = self._source_db.get_processed_db_source()[subject]

            if self._equalize_labels:
                while _same_number_of_labels(db_dict):
                    _reduce(db_dict)

                ep_num = len(db_dict[list(db_dict)[0]])
                kf = KFold(n_splits=self._k_fold_num)
                # todo: remove labels or not?
                for train_ind, test_ind in kf.split(np.arange(ep_num)):

            else:
                raise NotImplementedError

            # todo:
            #  - equalize labels
            #  - split to train test and validation
            #  - create lists for each set
            #  - shuffle data

    def split_db(self):
        """Splits database by subject. Use it at cross subject cross validation."""
        # raise NotImplementedError
        subjects = self._subject_db.get_subjects()

        if self._shuffle_subj:
            np.random.shuffle(subjects)

        if self._k_fold_num is not None:
            self._k_fold_num = max(1, self._k_fold_num)
            subjects = subjects[:self._k_fold_num]

        for s in subjects:
            yield subject_db.get_split(s, shuffle=self._shuffle_data)

    def split_subject_data_old(self, subject_db, subject_id):
        """Splits one subjects data to train and test set.

        Parameters
        ----------
        subject_db : OfflineDataPreprocessor
            The database from the subject data is selected.
        subject_id : int
            Subject ID number

        """
        if self._k_fold_num is None:
            self._k_fold_num = 10

        ep_list = subject_db.get_data_for_subject_split(subject_id)

        if subject_db.is_name('Physionet'):
            np.random.shuffle(ep_list)

        kf = KFold(n_splits=self._k_fold_num)
        for train_ind, test_ind in kf.split(list(range(len(ep_list)))):
            train = [ep_list[ind] for ind in train_ind]
            train = _generate_window_list_from_epoch_list(train)
            test = [ep_list[ind] for ind in test_ind]
            test = _generate_window_list_from_epoch_list(test)

            train = list(zip(*_reduce_max_label(*zip(*train))))
            test = list(zip(*_reduce_max_label(*zip(*test))))

            if self._shuffle_data:
                np.random.shuffle(train)
                np.random.shuffle(test)
                while self._over_max_follow(train):
                    np.random.shuffle(train)

            train_x, train_y = zip(*train)
            test_x, test_y = zip(*test)

            yield np.array(train_x), train_y, np.array(test_x), test_y


class OfflineDataPreprocessor:

    def __init__(self, base_dir, epoch_tmin=0, epoch_tmax=4, window_length=1.0, window_step=0.1,
                 use_drop_subject_list=True, fast_load=True, subject=None, select_eeg_file=False,
                 eeg_file=None):
        """Preprocessor for eeg files.

        Creates a database, which has all the required information about the eeg files.

        Parameters
        ----------
        base_dir : str
            Path for master database folder.
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
        subject : int, list of int, None
            ID of selected subjects.
        select_eeg_file : bool
            To select or not an eeg file.
        eeg_file : str, optional
            Absolute path to EEG file.
        """
        self._base_dir = Path(base_dir)
        self._epoch_tmin = epoch_tmin
        self._epoch_tmax = epoch_tmax  # seconds
        self._window_length = window_length  # seconds
        self._window_step = window_step  # seconds
        self._fast_load = fast_load
        self._subject_list = [subject] if type(subject) is int else subject
        self._select_one_file = select_eeg_file
        self.eeg_file = eeg_file

        self.info = mne.Info()
        self.feature_type = FeatureType.FFT_RANGE
        self.feature_kwargs = dict()

        self._data_path = Path()
        self._db_type = None  # Physionet / TTK

        self.proc_db_path = Path()
        self._proc_db_filenames = dict()
        self._proc_db_source = str()

        self._drop_subject = set() if use_drop_subject_list else None

    def use_db(self, db_name):
        if db_name == Databases.PHYSIONET:
            self.use_physionet()
        elif db_name == Databases.PILOT_PAR_A:
            self.use_pilot_par_a()
        elif db_name == Databases.PILOT_PAR_B:
            self.use_pilot_par_b()
        elif db_name == Databases.TTK:
            self.use_ttk_db()
        elif db_name == Databases.GAME:
            self.use_game_data()
        elif db_name == Databases.GAME_PAR_C:
            self.use_game_par_c()
        elif db_name == Databases.GAME_PAR_D:
            self.use_game_par_d()

        else:
            raise NotImplementedError('Database processor for {} db is not implemented'.format(db_name))
        return self

    def _use_db(self, db_type):
        """Loads a specified database."""
        self._data_path = self._base_dir.joinpath(db_type.DIR)
        assert self._data_path.exists(), "Path {} does not exists.".format(self._data_path)
        self._db_type = db_type

        if self._drop_subject is not None:
            self._drop_subject = set(db_type.DROP_SUBJECTS)
        else:
            self._drop_subject = set()

    def use_physionet(self):
        self._use_db(Physionet)
        return self

    def use_pilot_par_a(self):
        self._use_db(PilotDB_ParadigmA)
        return self

    def use_pilot_par_b(self):
        self._use_db(PilotDB_ParadigmB)
        return self

    def use_ttk_db(self):
        self._use_db(TTK_DB)
        return self

    def use_game_data(self):
        self._use_db(GameDB)
        return self

    def use_game_par_c(self):
        self._use_db(Game_ParadigmC)
        return self

    def use_game_par_d(self):
        self._use_db(Game_ParadigmD)
        return self

    @property
    def fs(self):
        return self.info['sfreq']

    def generate_mne_epoch(self, data):
        """Generates mne.Epoch from 3D array.

        Parameters
        ----------
        data : np.ndarray
            EEG data. shape should be like (n_epoch, n_channel, n_time_points)

        Returns
        -------
        mne.Epochs
        """
        return mne.EpochsArray(data, self.info)

    def run(self, feature_type=FeatureType.FFT_RANGE, **feature_kwargs):
        """Runs the Database preprocessor with the given features.

        Parameters
        ----------
        feature_type : FeatureType
            Specify the features which will be created in the preprocessing phase.
        feature_kwargs
            Arbitrary keyword arguments for feature extraction.
        """
        self.feature_type = feature_type
        self.feature_kwargs = feature_kwargs
        self._create_db()

    """
    Database functions
    """

    def _convert_task(self, record_number=None):
        if record_number is None:
            return self._db_type.TRIGGER_TASK_CONVERTER
        return self._db_type.TRIGGER_CONV_REC_TO_TASK.get(record_number)

    def get_subjects(self):
        return list(self._proc_db_filenames)

    def get_processed_db_source(self):
        return self._proc_db_filenames

    def get_labels(self):
        subj = list(self._proc_db_filenames)[0]
        return list(self._proc_db_filenames[subj])

    def get_command_converter(self):
        attr = 'COMMAND_CONV'
        assert hasattr(self._db_type, attr), '{} has no {} attribute'.format(self._db_type, attr)
        return self._db_type.COMMAND_CONV

    def is_name(self, db_name):
        return db_name in str(self._db_type)

    def _save_preprocessed_subject_data(self, subject_data, subj):
        """Saving preprocessed feature data for a given subject.

        Parameters
        ----------
        subject_data : dict
            Featured and preprocessed data.
        subj : int
            Subject number
        """
        print('Database generated for subject{}'.format(subj))
        subject_dict = dict()
        feature_ind = 0
        for task, ep_dict in subject_data.items():
            ep_file_dict = dict()
            for ind, ep_list in ep_dict.items():
                win_file_list = list()
                for ep in ep_list:
                    db_file = 'feature{}.data'.format(feature_ind)
                    save_pickle_data(str(self.proc_db_path.joinpath(db_file)), ep)
                    win_file_list.append(db_file)
                    feature_ind += 1
                ep_file_dict[ind] = win_file_list
            subject_dict[task] = ep_file_dict

        self._proc_db_filenames[subj] = subject_dict
        # processed db data...
        source = {
            SUBJECTS: self._get_subject_num(),
            DATA_SOURCE: self._proc_db_filenames,
            INFO: self.info
        }
        save_pickle_data(self._proc_db_source, source)

    def _get_subject_num(self):
        """Returns the number of available subjects in Database"""
        if self._db_type is Physionet:
            return self._db_type.SUBJECT_NUM

        file = 'rec01.vhdr'
        subject_num = len(sorted(Path(self._data_path).rglob(file)))

        return subject_num

    def _get_subject_list(self):  # todo: rethink...
        """Returns the list of subjects and removes the unwanted ones."""
        subject_num = self._get_subject_num()
        if self._subject_list is not None:
            for subj in self._subject_list:
                assert 0 < subj <= subject_num, 'Subject{} is not in subject range: 1 - {}'.format(
                    subj, subject_num)
            subject_list = self._subject_list
        else:
            subject_list = list(range(1, subject_num + 1))

        for subj in self._drop_subject:
            if subj in subject_list:
                subject_list.remove(subj)
                print('Dropping subject {}'.format(subj))
        return subject_list

    def _create_physionet_db(self):  # todo: rethink...
        """Physionet feature db creator function"""

        keys = self._db_type.TASK_TO_REC.keys()

        for subj in self._get_subject_list():

            subject_data = dict()

            for task in keys:
                recs = self._db_type.TASK_TO_REC.get(task)
                task_dict = self._convert_task(recs[0])
                filenames = generate_physionet_filenames(str(self._data_path.joinpath(self._db_type.FILE_PATH)), subj,
                                                         recs)
                epochs = get_epochs_from_files(filenames, task_dict, self._epoch_tmin, self._epoch_tmax)
                self.info = epochs.info
                win_epochs = self._get_windowed_features(epochs, task)

                subject_data.update(win_epochs)

            self._save_preprocessed_subject_data(subject_data, subj)

    def _create_ttk_db(self):
        """Pilot feature db creator function"""
        task_dict = self._convert_task()

        for subj in self._get_subject_list():
            if self._db_type is TTK_DB:
                filenames = generate_ttk_filenames(str(self._data_path.joinpath(self._db_type.FILE_PATH)), subj)
            else:
                filenames = generate_pilot_filenames(str(self._data_path.joinpath(self._db_type.FILE_PATH)), subj)

            cut_real_mov = REST in task_dict
            epochs = get_epochs_from_files(filenames, task_dict,
                                           self._epoch_tmin, self._epoch_tmax,
                                           cut_real_movement_tasks=cut_real_mov)
            self.info = epochs.info
            subject_data = self._get_windowed_features(epochs)
            self._save_preprocessed_subject_data(subject_data, subj)

    def _create_db_from_file(self):
        """Game database creation"""
        if self.eeg_file is None:
            self.eeg_file = select_file_in_explorer(str(self._base_dir))

        task_dict = self._convert_task()
        cut_real_mov = REST in task_dict
        epochs = get_epochs_from_files(self.eeg_file, task_dict,
                                       self._epoch_tmin, self._epoch_tmax,
                                       cut_real_movement_tasks=cut_real_mov)
        self.info = epochs.info
        subject_data = self._get_windowed_features(epochs)
        self._save_preprocessed_subject_data(subject_data, 0)

    def _get_windowed_features(self, epochs, task=None):
        """Feature creation from windowed data.

        self._feature : {'spatial', 'avg_column', 'column'}, optional
            The feature which will be created.

        Parameters
        ----------
        epochs : mne.Epochs
            Mne epochs from which the feature will be created.
        task : str, optional
            Selected from epochs.

        Returns
        -------
        dict
            Windowed feature data. dict of dict of list:
            task -> epoch -> windowed features

        """
        if task is not None:
            epochs = epochs[task]

        epochs.load_data()
        if self.feature_type is FeatureType.SPATIAL_TEMPORAL:
            epochs.resample(128, n_jobs=-1)

        task_list = [list(epochs[i].event_id.keys())[0] for i in range(len(epochs.selection))]

        window_length = self._window_length - 1 / self.fs  # win length correction
        win_num = int((self._epoch_tmax - self._epoch_tmin - window_length) / self._window_step) \
            if self._window_step > 0 else 1

        feature_extractor = FeatureExtractor(self.feature_type, self.fs, info=self.info, **self.feature_kwargs)

        task_dict = dict()
        for task in task_list:
            tsk_ep = epochs[task]
            win_epochs = {i: list() for i in range(len(tsk_ep))}
            for i in range(win_num):
                ep = tsk_ep.copy()
                ep.crop(ep.tmin + i * self._window_step, ep.tmin + window_length + i * self._window_step)
                feature = feature_extractor.run(ep.get_data())
                for j in range(len(tsk_ep)):
                    win_epochs[j].append((feature[j], task))
            task_dict[task] = win_epochs

        return task_dict

    def init_processed_db_path(self, feature=None):
        """Initialize the path of preprocessed database.

        Parameters
        ----------
        feature : Features, optional
            Feature used in preprocess.
        """
        if feature is not None and feature not in FeatureType:
            raise NotImplementedError('Feature {} is not implemented'.format(feature))
        if feature in FeatureType:
            self.feature_type = feature
        self.proc_db_path = self._base_dir.joinpath(DIR_FEATURE_DB, self._db_type.DIR, self.feature_type.name,
                                                    str(self._window_length), str(self._window_step))

    def _create_db(self):
        """Base db creator function."""

        assert self._db_type is not None, \
            'Define a database with .use_<db_name>() function before creating the database!'
        tic = time()
        self.init_processed_db_path()
        self._proc_db_source = str(self.proc_db_path.joinpath(self.feature_type.name + '.db'))
        Path(self.proc_db_path).mkdir(parents=True, exist_ok=True)

        def print_creation_message():
            print('{} file is not found. Creating database.'.format(file))

        file = self._proc_db_source
        fastload_source = load_pickle_data(file)
        n_subjects = self._get_subject_num()

        if fastload_source is not None and self._fast_load and \
                n_subjects == fastload_source[SUBJECTS]:
            subject_list = self._subject_list if self._subject_list is not None else np.arange(n_subjects) + 1

            self._proc_db_filenames = fastload_source[DATA_SOURCE]
            if all(subj in fastload_source[DATA_SOURCE] for subj in subject_list):
                self.info = fastload_source[INFO]
                return  # fast load ok. Do not create database.
            # extend existing fast load database.

        if self._db_type is Physionet:
            if self._select_one_file or self.eeg_file is not None:
                raise NotImplementedError('EEG file selection for Physionet is not implemented!')
            print_creation_message()
            self._create_physionet_db()

        elif not self._select_one_file and self.eeg_file is None:
            print_creation_message()
            self._create_ttk_db()

        elif self._db_type in [GameDB, Game_ParadigmC, Game_ParadigmD]:
            print_creation_message()
            self._create_db_from_file()

        else:
            raise NotImplementedError('Cannot create subject database for {}'.format(self._db_type))

        print('Database initialization took {} seconds.'.format(int(time() - tic)))

    # def get_subject_data(self, subject, reduce_rest=True, shuffle=True, random_seed=None):
    #     """Returns data for one subject.
    #
    #     Parameters
    #     ----------
    #     subject : int
    #         Number of the required subject.
    #     reduce_rest : bool, optional
    #         To reduce rest data points.
    #     """
    #     # todo: rethink reduce_rest --> reduce_max or reduce to min...
    #     assert subject not in self._drop_subject, 'Subject{} is in drop subject list.'.format(subject)
    #     assert subject in list(self._proc_db_filenames.keys()), \
    #         'Subject{} is not in preprocessed database'.format(subject)
    #     data, label = zip(*self._get_subject_eeg_data(subject))
    #     if reduce_rest:
    #         data, label = _reduce_max_label(data, label)
    #     if shuffle:
    #         ind = np.arange(len(label))
    #         np.random.seed(random_seed)
    #         np.random.shuffle(ind)
    #         data = [data[i] for i in ind]
    #         label = [label[i] for i in ind]
    #
    #     return list(data), list(label)

    # def _get_epoch_list(self, subject_id):
    #     ep_dict = self._data_set.get(subject_id)
    #     return list(ep_dict.values())

    # def _get_subject_eeg_data(self, subject_id):
    #     ep_list = self._get_epoch_list(subject_id)
    #     return _generate_window_list_from_epoch_list(ep_list)

    # def get_data_for_subject_split(self, subject_id):
    #     return self._get_epoch_list(subject_id)

    # def get_split(self, test_subject, shuffle=True, reduce_rest=True):
    #     """Splits the whole database to train and test sets.
    #
    #     This is a helper function for :class:`SubjectKFold` to make a split from the whole database.
    #     This function creates a train and test set. The test set is the data of the given subject
    #     and the train set is the rest of the database.
    #
    #     Parameters
    #     ----------
    #     test_subject : int
    #         Subject to put into the train set.
    #     shuffle : bool, optional
    #         Shuffle the train set if True
    #     random_seed : int, optional
    #         Random seed for shuffle.
    #     reduce_rest : bool, optional
    #         Make rest number equal to the max of other labels.
    #
    #     Returns
    #     -------
    #
    #     """
    #     train_subjects = list(self._data_set.keys())
    #     train_subjects.remove(test_subject)
    #
    #     train = list()
    #     test = self._get_subject_eeg_data(test_subject)
    #
    #     for s in train_subjects:
    #         train.extend(self._get_subject_eeg_data(s))
    #
    #     if shuffle:
    #         np.random.shuffle(train)
    #         np.random.shuffle(test)
    #
    #     train_x, train_y = zip(*train)
    #     test_x, test_y = zip(*test)
    #
    #     if reduce_rest:
    #         train_x, train_y = _reduce_max_label(train_x, train_y)
    #         test_x, test_y = _reduce_max_label(test_x, test_y)
    #
    #     return list(train_x), list(train_y), list(test_x), list(test_y), test_subject


if __name__ == '__main__':
    base_dir = init_base_config('..')
    epoch_proc = OfflineDataPreprocessor(base_dir, subject=1, fast_load=False)
    epoch_proc.use_game_par_d()
    feature_extraction = dict(
        feature_type=FeatureType.FFT_POWER,
        fft_low=14, fft_high=30
    )
    epoch_proc.run(**feature_extraction)
    print(epoch_proc.get_processed_db_source())

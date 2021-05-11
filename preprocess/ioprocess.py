from copy import deepcopy
from enum import Enum
from json import dump as json_dump, load as json_load
from pathlib import Path
from pickle import dump as pkl_dump, load as pkl_load
from sys import platform
from time import time

import mne
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

import preprocess.artefact_faster as art
from config import Physionet, PilotDB_ParadigmA, PilotDB_ParadigmB, TTK_DB, GameDB, Game_ParadigmC, Game_ParadigmD, \
    DIR_FEATURE_DB, REST, CALM, ACTIVE, BciCompIV1, BciCompIV2a, BciCompIV2b, ParadigmC
from gui_handler import select_file_in_explorer
from preprocess.feature_extraction import FeatureType, FeatureExtractor

EPOCH_DB = 'preprocessed_database'

# config options
CONFIG_FILE = 'bci_system.cfg'
BASE_DIR = 'base_dir'


class SourceDB(Enum):
    SUBJECTS = 'subjects'
    DATA_SOURCE = 'data_source'
    INFO = 'epoch_info'
    FEATURE_SHAPE = 'feature_shape'
    EPOCH_RANGE = 'epoch_range'
    MIMIC_ONLINE = 'mimic_online'


# db selection options
class Databases(Enum):
    PHYSIONET = 'physionet'
    PILOT_PAR_A = 'pilot_par_a'
    PILOT_PAR_B = 'pilot_par_b'
    TTK = 'ttk'
    GAME = 'game'
    GAME_PAR_C = 'game_par_c'
    GAME_PAR_D = 'game_par_d'
    BCI_COMP_IV_1 = 'BCICompIV1'
    BCI_COMP_IV_2A = 'BCICompIV2a'
    BCI_COMP_IV_2B = 'BCICompIV2b'
    ParadigmC = 'par_c'


def is_platform(os_platform):
    if 'win' in os_platform:
        os_platform = 'win'
    return platform.startswith(os_platform)


def open_raw_with_gui():
    return mne.io.read_raw(select_file_in_explorer(init_base_config()))


def _generate_filenames_for_subject(file_path, subject, subject_format_str, runs=1, run_format_str=None):
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

    Yields
    -------
    str
        filename for a subject with given runs

    """
    if type(runs) is not list:
        runs = [runs]

    for i in runs:
        f = file_path
        f = f.replace('{subj}', subject_format_str.format(subject))
        if run_format_str is not None:
            f = f.replace('{rec}', run_format_str.format(i))
        yield f


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
    rec : generator
        filenames for a subject with given runs

    """
    return _generate_filenames_for_subject(file_path, subject, '{:03d}', runs, '{:02d}')


def generate_pilot_filename(file_path, subject, runs=1):
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
        rec : generator
            filenames for a subject with given runs

        """
    return _generate_filenames_for_subject(file_path, subject, '{}', runs, '{:02d}')


def generate_ttk_filename(file_path, subject, runs=1):
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
        rec : generator
            filenames for a subject with given runs

        """
    return _generate_filenames_for_subject(file_path, subject, '{:02d}', runs, '{:02d}')


def generate_bci_comp_4_2a_filename(file_path, subject):
    return _generate_filenames_for_subject(file_path, subject, '{:02d}')


def load_pickle_data(filename):
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
    try:
        cfg_dict = load_pickle_data(file)
        base_directory = cfg_dict[BASE_DIR]
    except FileNotFoundError:
        from gui_handler import select_base_dir
        base_directory = select_base_dir()
        cfg_dict = {BASE_DIR: base_directory}
        save_pickle_data(file, cfg_dict)
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


def standardize_channel_names(raw):
    """Standardize channel positions and names.

    Specially designed for EEG channel standardization.
    source: mne.datasets.eegbci.standardize

    Parameters
    ----------
    raw : instance of Raw
        The raw data to standardize. Operates in-place.
    """
    rename = dict()
    for name in raw.ch_names:
        std_name = name.strip('.')
        std_name = std_name.upper()
        if std_name.endswith('Z'):
            std_name = std_name[:-1] + 'z'
        if 'FP' in std_name:
            std_name = std_name.replace('FP', 'Fp')
        if std_name.endswith('H'):
            std_name = std_name[:-1] + 'h'
        rename[name] = std_name
    raw.rename_channels(rename)


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
        Created epochs from files.
    """
    events, _ = mne.events_from_annotations(raw, event_id)
    # baseline = tuple([None, epoch_tmin + 0.1])  # if self._epoch_tmin > 0 else (None, 0)
    epochs = mne.Epochs(raw, events, baseline=baseline, event_id=task_dict, tmin=epoch_tmin,
                        tmax=epoch_tmax, preload=preload, on_missing='warning')
    return epochs


def get_epochs_from_files(filenames, task_dict, epoch_tmin=-0.2, epoch_tmax=0.5, baseline=None, event_id='auto',
                          preload=False, prefilter_signal=False, f_type='butter', order=5, l_freq=1, h_freq=None,
                          cut_real_movement_tasks=False):
    """Generate epochs from files.

    Parameters
    ----------
    filenames : str, list of str, generator
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
    raw = mne.io.concatenate_raws([mne.io.read_raw(file, preload=False) for file in filenames])

    standardize_channel_names(raw)
    try:  # check available channel positions
        mne.channels.make_eeg_layout(raw.info)
    except RuntimeError:  # if no channel positions are available create them from standard positions
        montage = mne.channels.make_standard_montage('standard_1005')  # 'standard_1020'
        raw.set_montage(montage, on_missing='warn')

    if cut_real_movement_tasks:
        raw = _cut_real_movemet_data(raw)

    if prefilter_signal:
        raw.load_data()
        iir_params = dict(order=order, ftype=f_type, output='sos')
        raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir', iir_params=iir_params, skip_by_annotation='edge')

    epochs = get_epochs_from_raw(raw, task_dict, epoch_tmin, epoch_tmax, baseline, event_id, preload=preload)

    return epochs


"""
Database dict manipulation functions
"""


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


def _do_label_equalization(db_dict):
    while not _same_number_of_labels(db_dict):
        _reduce(db_dict)


def _check_data_equality(data_dict):
    lens = [len(d_list) for d_list in data_dict.values()]
    first_len = lens[0]
    for ln in lens:
        assert ln == first_len, 'Number of data are not equal in each task.'


def _create_binary_db(task_dict):
    bin_dict = {_create_binary_label(label): dict() for label in task_dict}
    ep_count = {label: 0 for label in bin_dict}
    for label, ep_dict in task_dict.items():
        for ind, win_list in ep_dict.items():
            bin_dict[_create_binary_label(label)][ep_count[_create_binary_label(label)]] = win_list
            ep_count[_create_binary_label(label)] += 1
    return bin_dict


def _remove_subject_tag(db_dict, subject_list):
    db = {task: dict() for task in db_dict[subject_list[0]]}
    task_ind = {task: 0 for task in db_dict[subject_list[0]]}
    for subj in subject_list:
        for task, ep_dict in db_dict[subj].items():
            for ep in ep_dict.values():
                db[task][task_ind[task]] = ep
                task_ind[task] += 1
    return db


def _remove_task_tag(db_dict):
    no_task_db = dict()
    n = max([len(ep_dict) for ep_dict in db_dict.values()])
    ind = 0
    for i in range(n):
        for task in db_dict:
            ep = db_dict[task].pop(i, None)
            if ep is not None:
                no_task_db[ind] = ep
                ind += 1
    return no_task_db


def _get_file_list(db_dict, indexes):
    # return [file for i in indexes for file in db_dict[i]]
    return [db_dict[i] for i in indexes]


def _generate_file_db(source_db, equalize_labels=True, create_binary_db=False):
    source_db = deepcopy(source_db)
    if create_binary_db:
        source_db = _create_binary_db(source_db)
    if equalize_labels:
        _do_label_equalization(source_db)
    source_db = _remove_task_tag(source_db)
    return _get_file_list(source_db, np.arange(len(source_db)))


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
    elif ParadigmC.DIR in filename:
        db_name = Databases.ParadigmC
    else:
        raise ValueError('No database defined with path {}'.format(filename))
    return db_name


"""
Feature generation methods for train and test
"""


def _create_binary_label(label):
    if label != REST:
        label = CALM if CALM in label else ACTIVE
    return label


class SubjectKFold(object):

    def __init__(self, source_db, k_fold_num=None, validation_split=0.0, shuffle_subjects=True, shuffle_data=True,
                 random_state=None, equalize_labels=True, binarize_db=False):
        """Class to split subject database to train and test set, in an N-fold cross validation manner.

        Parameters
        ----------
        source_db : OfflineDataPreprocessor, OnlineDaraPreprocessor
            Source database with preprocessed filenames.
        k_fold_num : int, optional
        validation_split : float
            How much of the train set should be used as validation data. value range: [0, 1]
        shuffle_subjects : bool
        shuffle_data : bool
        random_state : int
        """
        self._source_db = source_db
        self._k_fold_num = k_fold_num
        self._validation_split = max(0.0, min(validation_split, 1.0))
        self._shuffle_subj = shuffle_subjects
        self._shuffle_data = shuffle_data
        np.random.seed(random_state)
        self._equalize_labels = equalize_labels
        self._binarize_db = binarize_db

        self._mimic_online_method = type(source_db) is OnlineDataPreprocessor

    def _get_train_and_val_ind(self, train_ind):
        val_num = int(len(train_ind) * self._validation_split)
        if self._validation_split > 0 and val_num == 0:
            val_num = 1
        np.random.shuffle(train_ind)
        val_ind = train_ind[:val_num]
        tr_ind = train_ind[val_num:]
        return tr_ind, val_ind

    def _split_one_versus_rest(self, subject, db_dict, subject_list):
        train_subj = subject_list.copy()
        test_subj = train_subj.pop(train_subj.index(subject))

        if self._k_fold_num is not None:
            k_fold_num = max(1, self._k_fold_num - 1)
            train_subj = train_subj[:k_fold_num]

        test_db = db_dict[test_subj]
        train_db = _remove_subject_tag(db_dict, train_subj)

        if self._validation_split > 0:  # creating validation set from train set not from one subject
            if self._binarize_db:
                train_db = _create_binary_db(train_db)
            if self._equalize_labels:
                _do_label_equalization(train_db)
            train_files = list()
            val_files = list()
            for task in list(train_db):
                train_ind = list(train_db[task])
                tr_ind, val_ind = self._get_train_and_val_ind(train_ind)
                val_files.extend(_get_file_list(train_db[task], val_ind))
                train_files.extend(_get_file_list(train_db[task], tr_ind))

            # train_files = np.array(train_files)
            # val_files = np.array(val_files)

            if self._shuffle_data:
                np.random.shuffle(train_files)
                np.random.shuffle(val_files)
        else:
            val_files = None
            train_files = _generate_file_db(train_db, self._equalize_labels, self._binarize_db)

        test_files = _generate_file_db(test_db, self._equalize_labels, self._binarize_db)
        return train_files, test_files, val_files, subject

    def _split_cross_subjects(self, subject):
        db_dict = self._source_db.get_processed_db_source(None)
        subject_list = list(db_dict)

        if self._shuffle_subj:
            np.random.shuffle(subject_list)

        if subject is not None:
            yield self._split_one_versus_rest(subject, db_dict, subject_list)
        else:
            if self._k_fold_num is not None:
                k_fold_num = max(1, self._k_fold_num)
                subject_list = subject_list[:k_fold_num]
                self._k_fold_num = None  # skipp this in _split_one_versus_rest()

            for subj in subject_list:
                yield self._split_one_versus_rest(subj, db_dict, subject_list)

    def _split_subject(self, subject):
        db_dict = self._source_db.get_processed_db_source(subject)
        if self._k_fold_num is None:
            self._k_fold_num = 5
        if self._binarize_db:
            db_dict = _create_binary_db(db_dict)
        if self._equalize_labels:
            _do_label_equalization(db_dict)

        kf_dict = {
            task: KFold(n_splits=self._k_fold_num, shuffle=self._shuffle_data).split(np.arange(len(epoch_dict))) for
            task, epoch_dict in db_dict.items()
        }

        for _ in range(self._k_fold_num):
            train_files = list()
            test_files = list()
            val_files = list()

            for task, epoch_dict in db_dict.items():
                train_ind, test_ind = next(kf_dict[task])
                test_files.extend(_get_file_list(epoch_dict, test_ind))
                if self._validation_split == 0:
                    train_files.extend(_get_file_list(epoch_dict, train_ind))
                    val_files = None
                else:
                    tr_ind, val_ind = self._get_train_and_val_ind(train_ind)
                    val_files.extend(_get_file_list(epoch_dict, val_ind))
                    train_files.extend(_get_file_list(epoch_dict, tr_ind))

            train_files = np.array(train_files)
            test_files = np.array(test_files)
            val_files = np.array(val_files) if val_files is not None else None

            if self._shuffle_data:
                np.random.shuffle(train_files)
                np.random.shuffle(test_files)
                if val_files is not None:
                    np.random.shuffle(val_files)

            yield train_files, test_files, val_files, subject

    def _split_online_subject_db(self, subject):
        cross_val_list = self._source_db.get_processed_db_source(subject)
        for train_db, test_db in cross_val_list:
            test_files = np.array(_generate_file_db(test_db, self._equalize_labels, self._binarize_db))
            if self._validation_split == 0:
                train_files = np.array(_generate_file_db(train_db, self._equalize_labels, self._binarize_db))
                val_files = None
            else:
                if self._binarize_db:
                    train_db = _create_binary_db(train_db)
                if self._equalize_labels:
                    _do_label_equalization(train_db)

                train_files = list()
                val_files = list()
                for task, epoch_dict in train_db.items():
                    tr_ind, val_ind = self._get_train_and_val_ind(list(epoch_dict))
                    val_files.extend(_get_file_list(epoch_dict, val_ind))
                    train_files.extend(_get_file_list(epoch_dict, tr_ind))

                train_files = np.array(train_files)
                val_files = np.array(val_files)

            yield train_files, test_files, val_files, subject

    def split(self, subject=None, cross_subject=False):
        """Split database to train and test sets in k-fold manner.

        Parameters
        ----------
        subject : int or None
            If cross subject is True the whole database will tested against
            the defined subject.
        cross_subject : bool
            Split the data in a cross-subject fashion.
        """
        if not self._mimic_online_method:
            if cross_subject:
                return self._split_cross_subjects(subject)
            else:
                assert subject is not None, 'Subject must be defined for subject split!'
                return self._split_subject(subject)

        else:  # mimic_online_method
            if cross_subject:
                raise NotImplementedError('Cross-subject split is not implemeted for online mimic method.')

            assert subject is not None, 'Subject must be defined for subject split!'
            return self._split_online_subject_db(subject)


class DataLoader:

    def __init__(self, base_dir, use_drop_subject_list=True):
        """Data loader

        Helper class for loading different databases from HardDrive.

        Parameters
        ----------
        base_dir : str
            Path for master database folder.
        use_drop_subject_list : bool
            Whether to use drop subject list from config file or not?
        """
        self._base_dir = Path(base_dir)
        self.info = mne.Info()

        self._data_path = Path()
        self._db_type = None  # Physionet / TTK / ect...

        self._subject_list = None
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
        elif db_name == Databases.BCI_COMP_IV_1:
            self.use_bci_comp_4_1()
        elif db_name == Databases.BCI_COMP_IV_2A:
            self.use_bci_comp_4_2a()
        elif db_name == Databases.BCI_COMP_IV_2B:
            self.use_bci_comp_4_2b()
        elif db_name == Databases.ParadigmC:
            self.use_par_c()

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

    def get_data_path(self):
        assert self._db_type is not None, 'Database is not defined.'
        return self._data_path

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

    def use_bci_comp_4_1(self):
        self._use_db(BciCompIV1)
        return self

    def use_bci_comp_4_2a(self):
        self._use_db(BciCompIV2a)
        return self

    def use_bci_comp_4_2b(self):
        self._use_db(BciCompIV2b)
        return self

    def use_par_c(self):
        self._use_db(ParadigmC)
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
        if len(data.shape) == 2:
            data = np.expand_dims(data, 0)
        return mne.EpochsArray(data, self.info)

    def validate_make_binary_classification_use(self):
        assert self._db_type is not None, 'Database is not defined.'
        if self._db_type == Physionet and not Physionet.USE_NEW_CONFIG:
            if REST not in Physionet.TASK_TO_REC:
                raise ValueError(f'Can not make binary classification. Check values of '
                                 f'TASK_TO_REC in class {self._db_type.__name__} '
                                 f'in config.py file. REST should not be commented out!')
        elif REST not in self._db_type.TRIGGER_TASK_CONVERTER:
            implemented_dbs = [GameDB, Game_ParadigmC, ParadigmC, PilotDB_ParadigmA, PilotDB_ParadigmB,
                               TTK_DB, Physionet]
            not_implemented_dbs = [BciCompIV1, BciCompIV2a, BciCompIV2b]
            if self._db_type in implemented_dbs:
                raise ValueError(f'Can not make binary classification. Check values of '
                                 f'TRIGGER_TASK_CONVERTER in class {self._db_type.__name__} '
                                 f'in config.py file. REST should not be commented out!')
            elif self._db_type in not_implemented_dbs:
                raise NotImplementedError(f'{self._db_type.__name__} is not implemented for '
                                          f'make_binary_classification.\nYou can comment out some values '
                                          f'from the TRIGGER_TASK_CONVERTER in class {self._db_type.__name__} '
                                          f'in config.py file to make the classification binary.')
            elif self._db_type == Game_ParadigmD:
                pass  # always OK
            else:
                raise NotImplementedError(f'class {self._db_type.__name__} is not yet integrated...')

    def is_subject_in_drop_list(self, subject):
        return subject in self._drop_subject

    def _convert_task(self, record_number=None):
        if record_number is None:
            return self._db_type.TRIGGER_TASK_CONVERTER
        return self._db_type.TRIGGER_CONV_REC_TO_TASK.get(record_number)

    def get_command_converter(self):
        attr = 'COMMAND_CONV'
        assert hasattr(self._db_type, attr), '{} has no {} attribute'.format(self._db_type, attr)
        return self._db_type.COMMAND_CONV

    def is_name(self, db_name):
        return db_name in str(self._db_type)

    def get_subject_num(self):
        """Returns the number of available subjects in Database"""
        if self._db_type in [Physionet, BciCompIV1, BciCompIV2a, BciCompIV2b]:
            subject_num = self._db_type.SUBJECT_NUM
        elif self._db_type in [TTK_DB, PilotDB_ParadigmA, PilotDB_ParadigmB, Game_ParadigmC, Game_ParadigmD, ParadigmC]:
            file = 'rec01.vhdr'
            subject_num = len(sorted(Path(self._data_path).rglob(file)))
        else:
            raise NotImplementedError('get_subject_num is undefined for {}'.format(self._db_type))

        return subject_num

    def get_subject_list(self):  # todo: rethink...
        """Returns the list of subjects and removes the unwanted ones."""
        subject_num = self.get_subject_num()
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

    def get_file_path_for_db_filename_gen(self):
        assert self._db_type is not None, 'Database is not defined.'
        return str(self._data_path.joinpath(self._db_type.FILE_PATH))


class DataProcessor(DataLoader):
    def __init__(self, base_dir, epoch_tmin=0, epoch_tmax=4, window_length=1.0, window_step=0.1,
                 use_drop_subject_list=True, fast_load=False,
                 *,
                 filter_params=None, do_artefact_rejection=False, artefact_thresholds=None):
        """Abstract Preprocessor for eeg files.

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
        filter_params : dict, optional
            Parameters for Butterworth highpass digital filtering. ''order'' and ''l_freq''
        do_artefact_rejection : bool
            To do or not artefact-rejection
        """
        if filter_params is None:
            filter_params = {}

        self._epoch_tmin = epoch_tmin
        self._epoch_tmax = epoch_tmax  # seconds
        self._window_length = window_length  # seconds
        self._window_step = window_step  # seconds
        self._filter_params = filter_params

        self._fast_load = fast_load

        self.feature_type = FeatureType.FFT_RANGE
        self.feature_kwargs = dict()
        self._feature_shape = tuple()

        self.proc_db_path = Path()
        self._proc_db_filenames = dict()
        self._proc_db_source = str()

        self._do_artefact_rejection = do_artefact_rejection
        self._mimic_online_method = False

        if self._do_artefact_rejection:
            self.artefact_filter = art.ArtefactFilter(thresholds=artefact_thresholds, apply_frequency_filter=False)
        else:
            self.artefact_filter = None

        super(DataProcessor, self).__init__(base_dir, use_drop_subject_list)

    def _create_db(self):
        raise NotImplementedError

    def get_subjects(self):
        return list(self._proc_db_filenames)

    def get_processed_db_source(self):
        raise NotImplementedError

    def get_labels(self):
        raise NotImplementedError

    def get_feature_shape(self):
        return self._feature_shape

    def init_processed_db_path(self, feature=None):
        """Initialize the path of preprocessed database.

        Parameters
        ----------
        feature : FeatureType, optional
            Feature used in preprocess.
        """
        if feature is not None:
            if feature in FeatureType:
                self.feature_type = feature
            else:
                raise NotImplementedError('Feature {} is not implemented'.format(feature))
        feature_dir = self.feature_type.name + str(list(self.feature_kwargs.values())).replace(' ', '')
        filter_dir = str(list(self._filter_params.values())).replace(' ', '').replace('[', '').replace(']', '')

        if not is_platform('win') and Path(DIR_FEATURE_DB).is_absolute():
            self.proc_db_path = Path(DIR_FEATURE_DB)
        else:
            self.proc_db_path = self._base_dir.joinpath(DIR_FEATURE_DB)
        self.proc_db_path = self.proc_db_path.joinpath(self._db_type.DIR, feature_dir, filter_dir,
                                                       str(self._window_length), str(self._window_step))

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
        if task is not None:  # only for Physionet old config
            epochs = epochs[task]

        if self._do_artefact_rejection:
            if self._db_type == Physionet and not Physionet.USE_NEW_CONFIG:
                raise NotImplementedError('Artefact rejection is implemented for Physionet database '
                                          'with "USE_NEW_CONFIG = True". Change it in config.py')
            else:
                if self._mimic_online_method:
                    epochs = self.artefact_filter.online_filter(epochs)
                else:
                    epochs = self.artefact_filter.offline_filter(epochs)

        self.info = epochs.info
        epochs.load_data()

        if self.feature_type is FeatureType.SPATIAL_TEMPORAL:  # todo: move it after epoch corp?
            if self._db_type is not Databases.PHYSIONET:
                epochs.resample(62.5, n_jobs=-1)  # form 500 Hz / 8 --> maxf = 31.25 Hz
            else:
                raise NotImplementedError

        assert self._window_length <= self._epoch_tmax - self._epoch_tmin, \
            'Can not create {} sec long windows, because it is longer than the {} sec long epoch'.format(
                self._window_length, self._epoch_tmax - self._epoch_tmin)

        window_length = self._window_length - 1 / self.fs  # win length correction
        if self._window_step == 0 or self._window_length == self._epoch_tmax - self._epoch_tmin:
            win_num = 1
        else:
            win_num = int((self._epoch_tmax - self._epoch_tmin - window_length) / self._window_step)

        feature_extractor = FeatureExtractor(self.feature_type, self.fs, info=self.info, **self.feature_kwargs)

        task_set = set([list(epochs[i].event_id.keys())[0] for i in range(len(epochs.selection))])
        task_dict = dict()
        for task in task_set:
            tsk_ep = epochs[task]
            win_epochs = {i: list() for i in range(len(tsk_ep))}
            for i in range(win_num):  # cannot speed up here with parallel process...
                ep = tsk_ep.copy().pick('eeg')
                ep.crop(ep.tmin + i * self._window_step, ep.tmin + window_length + i * self._window_step)
                feature = feature_extractor.run(ep.get_data())
                f_shape = feature.shape[1:]
                if len(self._feature_shape) > 0:
                    assert f_shape == self._feature_shape, 'Error: Change in feature output shape. prev: {},  ' \
                                                           'current: {}'.format(self._feature_shape, f_shape)
                self._feature_shape = f_shape
                for j in range(len(tsk_ep)):
                    win_epochs[j].append((feature[j], np.array([task])))
            task_dict[task] = win_epochs

        return task_dict

    def _init_fast_load_data(self):
        self._proc_db_source = str(self.proc_db_path.joinpath(self.feature_type.name + '.db'))
        Path(self.proc_db_path).mkdir(parents=True, exist_ok=True)
        self._feature_shape = tuple()

        try:
            fastload_source = load_pickle_data(self._proc_db_source)
        except FileNotFoundError:
            fastload_source = None
        n_subjects = self.get_subject_num()

        if fastload_source is not None and self._fast_load and \
                len(fastload_source) == len(SourceDB) and \
                n_subjects == fastload_source[SourceDB.SUBJECTS] and \
                (self._epoch_tmin, self._epoch_tmax) == fastload_source[SourceDB.EPOCH_RANGE] and \
                fastload_source[SourceDB.MIMIC_ONLINE] == self._mimic_online_method:

            subject_list = self._subject_list if self._subject_list is not None else np.arange(n_subjects) + 1
            subject_list = [subject for subject in subject_list if subject not in self._drop_subject]

            self._proc_db_filenames = fastload_source[SourceDB.DATA_SOURCE]
            if all(subj in fastload_source[SourceDB.DATA_SOURCE] for subj in subject_list):
                self.info = fastload_source[SourceDB.INFO]
                self._feature_shape = fastload_source[SourceDB.FEATURE_SHAPE]
                return True  # fast load ok. Do not create database.
            # extend existing fast load database.
        return False

    def _save_fast_load_source_data(self):
        source = {
            SourceDB.SUBJECTS: self.get_subject_num(),
            SourceDB.DATA_SOURCE: self._proc_db_filenames,
            SourceDB.INFO: self.info,
            SourceDB.FEATURE_SHAPE: self._feature_shape,
            SourceDB.EPOCH_RANGE: (self._epoch_tmin, self._epoch_tmax),
            SourceDB.MIMIC_ONLINE: self._mimic_online_method
        }
        save_pickle_data(self._proc_db_source, source)

    def _parallel_generate_db(self, func):
        """Parallel DB generation for each subject"""
        subject_list = self.get_subject_list()

        if len(subject_list) > 1:
            data, info = zip(*Parallel(n_jobs=-2)(delayed(func)(subject) for subject in subject_list))
            self.info = info[0][0]
            self._feature_shape = info[0][1]
        else:
            data, _ = zip(*[func(subject) for subject in subject_list])
        self._proc_db_filenames.update(data)
        self._save_fast_load_source_data()

    def run(self, subject=None, feature_type=FeatureType.FFT_RANGE, **feature_kwargs):
        """Runs the Database preprocessor with the given features.

        Parameters
        ----------
        subject : int, list of int, None
            ID of selected subjects.
        feature_type : FeatureType
            Specify the features which will be created in the preprocessing phase.
        feature_kwargs
            Arbitrary keyword arguments for feature extraction.
        """
        self._subject_list = [subject] if type(subject) is int else subject
        self.feature_type = feature_type
        self.feature_kwargs = feature_kwargs

        assert self._db_type is not None, \
            'Define a database with .use_<db_name>() function before creating the database!'
        tic = time()
        self.init_processed_db_path()
        self._create_db()
        print('Database initialization took {} seconds.'.format(int(time() - tic)))


class OfflineDataPreprocessor(DataProcessor):

    def __init__(self, base_dir, epoch_tmin=0, epoch_tmax=4, window_length=1.0, window_step=0.1,
                 use_drop_subject_list=True, fast_load=True,
                 *,
                 select_eeg_file=False, eeg_file=None, filter_params=None,
                 do_artefact_rejection=False, artefact_thresholds=None):
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
        select_eeg_file : bool
            To select or not an eeg file.
        eeg_file : str, optional
            Absolute path to EEG file.
        filter_params : dict, optional
            Parameters for Butterworth highpass digital filtering. ''order'' and ''l_freq''
        do_artefact_rejection : bool
            To do or not artefact-rejection
        """
        self._eeg_file = eeg_file
        self._select_one_file = select_eeg_file

        super(OfflineDataPreprocessor, self).__init__(
            base_dir, epoch_tmin, epoch_tmax, window_length, window_step,
            use_drop_subject_list, fast_load,
            filter_params=filter_params,
            do_artefact_rejection=do_artefact_rejection, artefact_thresholds=artefact_thresholds
        )

    def get_labels(self, make_binary_classification=False):
        subj = list(self._proc_db_filenames)[0]
        label_list = list(self._proc_db_filenames[subj])
        if make_binary_classification:
            label_list = list(set([_create_binary_label(label) for label in label_list]))
        return label_list

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
        subject_file_dict = dict()
        epoch_ind = 0
        for task, ep_dict in subject_data.items():
            ep_file_dict = dict()
            for ind, ep_list in ep_dict.items():
                db_file = 'subj{}-epoch{}.data'.format(subj, epoch_ind)
                db_file = str(self.proc_db_path.joinpath(db_file))
                save_pickle_data(db_file, ep_list)
                epoch_ind += 1
                ep_file_dict[ind] = db_file
            subject_file_dict[task] = ep_file_dict
        return (subj, subject_file_dict), (self.info, self._feature_shape)

    def get_processed_db_source(self, subject=None, equalize_labels=True, only_files=False):
        if not only_files:  # SubjectKFold method
            if subject is None:
                return self._proc_db_filenames
            return self._proc_db_filenames[subject]
        else:
            if subject is None:  # online BCI method
                db_files = self._proc_db_filenames
                db_files = _remove_subject_tag(db_files, list(db_files))
            else:
                db_files = self._proc_db_filenames[subject]
            return _generate_file_db(db_files, equalize_labels)

    def _create_physionet_db(self, subj):
        subject_data = dict()
        keys = self._db_type.TASK_TO_REC.keys()

        for task in keys:
            recs = self._db_type.TASK_TO_REC.get(task)
            task_dict = self._convert_task(recs[0])
            filenames = generate_physionet_filenames(str(self._data_path.joinpath(self._db_type.FILE_PATH)), subj,
                                                     recs)
            epochs = get_epochs_from_files(filenames, task_dict, self._epoch_tmin, self._epoch_tmax,
                                           prefilter_signal=len(self._filter_params) > 0,
                                           **self._filter_params)
            self.info = epochs.info
            win_epochs = self._get_windowed_features(epochs, task)

            subject_data.update(win_epochs)
        return self._save_preprocessed_subject_data(subject_data, subj)

    def _create_x_db(self, subj):
        task_dict = self._convert_task()
        if not hasattr(self._db_type, 'USE_NEW_CONFIG') or not self._db_type.USE_NEW_CONFIG:
            cut_real_mov = REST in task_dict
            if self._db_type is TTK_DB:
                fn_gen = generate_ttk_filename(str(self._data_path.joinpath(self._db_type.FILE_PATH)), subj)
            elif self._db_type in [PilotDB_ParadigmA, PilotDB_ParadigmB, GameDB, Game_ParadigmC, Game_ParadigmD,
                                   ParadigmC]:
                fn_gen = generate_pilot_filename(str(self._data_path.joinpath(self._db_type.FILE_PATH)), subj)
            elif self._db_type in [BciCompIV1, BciCompIV2a]:
                fn_gen = generate_bci_comp_4_2a_filename(str(self._data_path.joinpath(self._db_type.FILE_PATH)), subj)
            elif self._db_type is BciCompIV2b:
                s_ind = subj - 1
                s = s_ind // 3 + 1
                rec = s_ind % 3 + 1

                fn_gen = generate_ttk_filename(str(self._data_path.joinpath(self._db_type.FILE_PATH)), s, rec)
            else:
                raise NotImplementedError('Database loading for {} is not implemented'.format(self._db_type))
        else:  # self._db_type.USE_NEW_CONFIG == True
            cut_real_mov = False
            if self._db_type in [Physionet, TTK_DB]:
                pattern = self._db_type.FILE_PATH
                pattern = pattern.replace('{subj}', '{:03d}'.format(subj))
                pattern = pattern.replace('{rec}', '*')
                fn_gen = list(sorted(self._data_path.rglob(pattern)))
                assert len(fn_gen) > 0, f'No files were found. Try to set USE_NEW_CONFIG=False ' \
                                        f'for {self._db_type} or download the leatest database.'
            else:
                raise NotImplementedError('Database loading for {} with USE_NEW_CONFIG=True '
                                          'is not implemented.'.format(self._db_type))

        epochs = get_epochs_from_files(fn_gen, task_dict,
                                       self._epoch_tmin, self._epoch_tmax,
                                       cut_real_movement_tasks=cut_real_mov,
                                       prefilter_signal=len(self._filter_params) > 0,
                                       event_id=self._db_type.TRIGGER_EVENT_ID,
                                       **self._filter_params)
        subject_data = self._get_windowed_features(epochs)
        return self._save_preprocessed_subject_data(subject_data, subj)

    def _create_db_from_file(self):
        """Game database creation"""
        if self._eeg_file is None:
            self._eeg_file = select_file_in_explorer(str(self._base_dir))

        task_dict = self._convert_task()
        cut_real_mov = REST in task_dict
        epochs = get_epochs_from_files(self._eeg_file, task_dict,
                                       self._epoch_tmin, self._epoch_tmax,
                                       cut_real_movement_tasks=cut_real_mov,
                                       prefilter_signal=len(self._filter_params) > 0,
                                       event_id=self._db_type.TRIGGER_EVENT_ID,
                                       **self._filter_params)
        subject_data = self._get_windowed_features(epochs)
        data, _ = zip(*[self._save_preprocessed_subject_data(subject_data, 0)])
        self._proc_db_filenames = dict(data)
        self._save_fast_load_source_data()

    def _create_db(self):
        """Base db creator function."""

        if self._init_fast_load_data():
            return  # fast load ok. Do not create database.
        print('{} file is not found. Creating database.'.format(self._proc_db_source))

        if self._db_type is Physionet and not Physionet.USE_NEW_CONFIG:
            if self._select_one_file or self._eeg_file is not None:
                raise NotImplementedError('EEG file selection for Physionet is not implemented!')
            self._parallel_generate_db(self._create_physionet_db)

        elif not self._select_one_file and self._eeg_file is None:
            self._parallel_generate_db(self._create_x_db)

        elif self._db_type in [GameDB, Game_ParadigmC, Game_ParadigmD, ParadigmC]:
            self._create_db_from_file()

        else:
            raise NotImplementedError('Cannot create subject database for {}'.format(self._db_type))

    def get_feature(self, subject=1, task=0, epoch=0, window=0):
        """Select a feature form processed Database

        Parameters
        ----------
        subject : int
            Subject number.
        task : str, int
            Number or name of the task in the database.
        epoch : int
            Number of epoch
        window : int
            Number of window in epoch

        Returns
        -------
        tuple
            Datapoint with label. (datapoint, label)
            Datapoint is an ndarray.
        """
        sdb = self.get_processed_db_source(subject)
        if type(task) is int:
            task = list(sdb)[task]
        elif type(task) is str:
            pass
        else:
            raise TypeError('Task type must be int or str.')
        filename = sdb[task][epoch]
        window_list = load_pickle_data(filename)
        return window_list[window]


class OnlineDataPreprocessor(DataProcessor):

    def __init__(self, base_dir, epoch_tmin=0, epoch_tmax=4, window_length=1.0, window_step=0.1,
                 use_drop_subject_list=True, fast_load=True,
                 *,
                 filter_params=None, do_artefact_rejection=True, artefact_thresholds=None,
                 n_fold=5, shuffle=True):
        """Online Data Preprocessor for eeg files.

        Mimic online data processing. Splitting data on Session level between
        train and test set.

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
        filter_params : dict, optional
            Parameters for Butterworth highpass digital filtering. ''order'' and ''l_freq''
        do_artefact_rejection : bool
            To do or not artefact-rejection
        n_fold : int
            Number of folds for N-fold-Xvalidation.
        shuffle : bool
            Whether to shuffle the data before splitting into batches in N-fold-Xvalidation.
        """
        assert n_fold is not None, 'n_fold must be an integer value greater than 1'
        self._n_fold = n_fold
        self._shuffle = shuffle

        super(OnlineDataPreprocessor, self).__init__(
            base_dir, epoch_tmin, epoch_tmax, window_length, window_step,
            use_drop_subject_list, fast_load,
            filter_params=filter_params,
            do_artefact_rejection=do_artefact_rejection, artefact_thresholds=artefact_thresholds
        )
        self._mimic_online_method = True

    def get_labels(self, make_binary_classification=False):
        subj = list(self._proc_db_filenames)[0]
        label_list = list(self._proc_db_filenames[subj][0][0])
        if make_binary_classification:
            label_list = list(set([_create_binary_label(label) for label in label_list]))
        return label_list

    def get_subject_num(self):
        """Returns the number of available subjects in Database"""
        if self._db_type in [Physionet, BciCompIV1, BciCompIV2a, BciCompIV2b]:
            subject_num = self._db_type.SUBJECT_NUM
        elif self._db_type in [TTK_DB, PilotDB_ParadigmA, PilotDB_ParadigmB, Game_ParadigmC, Game_ParadigmD, ParadigmC]:
            file = '*R01_raw.fif'
            subject_num = len(sorted(Path(self._data_path).rglob(file)))
        else:
            raise NotImplementedError('get_subject_num is undefined for {}'.format(self._db_type))

        return subject_num

    def _save_preprocessed_subject_data(self, subject_data, subj, xval_type):
        """Saving preprocessed feature data for a given subject.

        Parameters
        ----------
        subject_data : dict
            Featured and preprocessed data.
        subj : int
            Subject number
        """
        print('Database generated for subject{}'.format(subj))
        subject_file_dict = dict()
        epoch_ind = 0
        for task, ep_dict in subject_data.items():
            ep_file_dict = dict()
            for ind, ep_list in ep_dict.items():
                db_file = f'{xval_type}-subj{subj}-epoch{epoch_ind}.data'
                db_file = str(self.proc_db_path.joinpath(db_file))
                save_pickle_data(db_file, ep_list)
                epoch_ind += 1
                ep_file_dict[ind] = db_file
            subject_file_dict[task] = ep_file_dict
        return subject_file_dict

    def get_processed_db_source(self, subject=None):
        if subject is None:
            return self._proc_db_filenames
        return self._proc_db_filenames[subject]

    def generate_db_from_file_list(self, filenames):
        task_dict = self._convert_task()
        epochs = get_epochs_from_files(filenames, task_dict, self._epoch_tmin, self._epoch_tmax,
                                       prefilter_signal=len(self._filter_params) > 0,
                                       event_id=self._db_type.TRIGGER_EVENT_ID,
                                       **self._filter_params)
        return self._get_windowed_features(epochs)

    def _generate_online_db(self, subj):
        pattern = self._db_type.FILE_PATH
        pattern = pattern.replace('{subj}', '{:03d}'.format(subj))
        pattern = pattern.replace('{rec}', '*')
        session_files = np.array(sorted(self._data_path.rglob(pattern)))

        kfold_data = list()
        self._n_fold = min(len(session_files), self._n_fold)
        kfold = KFold(n_splits=self._n_fold, shuffle=self._shuffle)
        for i, (train_index, test_index) in enumerate(kfold.split(session_files)):
            self._mimic_online_method = False
            subject_train = self.generate_db_from_file_list(session_files[train_index])
            train_files = self._save_preprocessed_subject_data(subject_train, subj, f'train{i}')
            self._mimic_online_method = True
            subject_test = self.generate_db_from_file_list(session_files[test_index])
            test_files = self._save_preprocessed_subject_data(subject_test, subj, f'test{i}')

            kfold_data.append((train_files, test_files))

        return (subj, kfold_data), (self.info, self._feature_shape)

    def _create_db(self):
        if self._init_fast_load_data():
            return  # fast load ok. Do not create database.
        print('{} file is not found. Creating database.'.format(self._proc_db_source))

        if not hasattr(self._db_type, 'USE_NEW_CONFIG'):
            raise NotImplementedError(f'Online mimic process is only implemented for new config. '
                                      f'Please reformat class {self._db_type.__name__} '
                                      f'in the config.py file.')

        if not self._db_type.USE_NEW_CONFIG:
            raise NotImplementedError(f'Online mimic process is only implemented for new config. '
                                      f'Please set USE_NEW_CONFIG=True in the config.py file '
                                      f'at class {self._db_type.__name__}.')

        if self._db_type in [Physionet, TTK_DB]:
            self._parallel_generate_db(self._generate_online_db)
        else:
            raise NotImplementedError('Cannot create subject database for {}'.format(self._db_type))


class DataHandler:  # todo: move to TFRecord - https://www.tensorflow.org/guide/data#consuming_tfrecord_data

    def __init__(self, file_list, label_encoder, binary_classification=False,
                 shuffle_epochs=True, epoch_buffer=5, shuffle_windows=True):
        """Data handler for data, which is greater than the memory capacity.

        This data handler integrates the data with the tensorflow dataset API.

        Parameters
        ----------
        file_list : list of str
            list of epoch filenames
        label_encoder : LabelEncoder
            In case of binary classification converts the labels.
        binary_classification : bool
            To use or not the label converter.
        shuffle_epochs : bool
            If yes, the order of epochs will be shuffled.
        epoch_buffer : int
            How much epochs should be preloaded. Used for window shuffling.
        shuffle_windows : bool
            If yes, the order of windows will be shuffled in the epoch buffer.
        """
        if shuffle_epochs:
            np.random.shuffle(file_list)
        self._file_list = list(file_list)
        self._label_encoder = label_encoder
        self._binary_classification = binary_classification
        window_list = load_pickle_data(self._file_list[0])
        data, label = window_list[0]
        self._win_num = len(window_list)
        self._data_shape = data.shape
        self._label_shape = label.shape
        self._preload_epoch_num = epoch_buffer
        self._shuffle_windows = shuffle_windows
        self._data_list = list()

    def _load_epoch(self, filename):
        window_list = load_pickle_data(filename)
        for data, labels in window_list:
            if self._binary_classification:
                labels = [_create_binary_label(label) for label in labels]
            labels = self._label_encoder.transform(labels)
            self._data_list.append((data, labels))

    def _generate_data(self):
        for i in range(self._preload_epoch_num):
            self._load_epoch(self._file_list[i])
        for i in range(self._preload_epoch_num, len(self._file_list)):
            self._load_epoch(self._file_list[i])
            if self._shuffle_windows:
                np.random.shuffle(self._data_list)
            for _ in range(self._win_num):
                yield self._data_list.pop(0)

        while len(self._data_list) > 0:  # all files loaded
            yield self._data_list.pop(0)

    def get_tf_dataset(self):
        """Getting tensorflow Dataset from generator."""
        # dataset = tf.data.Dataset.from_tensor_slices(self._file_list)  # or generator
        # dataset = dataset.map(self._load_data)
        import tensorflow as tf
        dataset = tf.data.Dataset.from_generator(
            self._generate_data,
            output_types=(tf.float32, tf.int32),
            output_shapes=(self._data_shape, self._label_shape)
        )
        return dataset

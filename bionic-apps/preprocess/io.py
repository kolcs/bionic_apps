import mne
from pathlib import Path
from enum import Enum
import numpy as np

from ..utils import standardize_eeg_channel_names, init_base_config
from ..handlers.gui import select_files_in_explorer
from ..databases import EEG_Databases, get_eeg_db_name_by_filename, Physionet, TTK_DB, \
    PilotDB_ParadigmA, PilotDB_ParadigmB, GameDB, Game_ParadigmC, Game_ParadigmD, ParadigmC, EmotivParC, \
    BciCompIV2a, BciCompIV2b, BciCompIV1, Giga, REST


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
                        tmax=epoch_tmax, preload=preload, on_missing='warn')
    return epochs


def get_epochs_from_files(filenames, task_dict, epoch_tmin=-0.2, epoch_tmax=0.5, baseline=None, event_id='auto',
                          preload=False, prefilter_signal=False, f_type='butter', order=5, l_freq=1, h_freq=None):
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

    raw_list = list()
    for file in filenames:
        if Path(file).suffix == '.xdf':
            from ..external_connections.emotiv.mne_import_xdf import read_raw_xdf
            raw_list.append(read_raw_xdf(file))
        else:
            raw_list.append(mne.io.read_raw(file, preload=False))

    raw = mne.io.concatenate_raws(raw_list)

    standardize_eeg_channel_names(raw)
    try:  # check available channel positions
        mne.channels.make_eeg_layout(raw.info)
    except RuntimeError:  # if no channel positions are available create them from standard positions
        montage = mne.channels.make_standard_montage('standard_1005')  # 'standard_1020'
        raw.set_montage(montage, on_missing='warn')

    if prefilter_signal:
        raw.load_data()
        iir_params = dict(order=order, ftype=f_type, output='sos')
        raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir', iir_params=iir_params, skip_by_annotation='edge')

    epochs = get_epochs_from_raw(raw, task_dict, epoch_tmin, epoch_tmax, baseline, event_id, preload=preload)

    return epochs


def open_raw_with_gui():
    raws = [mne.io.read_raw(file) for file in sorted(select_files_in_explorer(init_base_config()))]
    raw = mne.io.concatenate_raws(raws)
    return raw


def get_epochs_from_raw_with_gui(epoch_tmin=0, epoch_tmax=4, baseline=(None, .1)):
    files = select_files_in_explorer(init_base_config())
    db_name = get_eeg_db_name_by_filename(files[0])
    loader = DataLoader().use_db(db_name)

    raws = [mne.io.read_raw(file) for file in files]
    raw = mne.io.concatenate_raws(raws)

    task_dict = loader.get_task_dict()
    event_id = loader.get_event_id()
    return get_epochs_from_raw(raw, task_dict, epoch_tmin, epoch_tmax, baseline, event_id)


class SubjectHandle(Enum):
    INDEPENDENT_DAYS = 1
    BCI_COMP = 2
    MIX_EXPERIMENTS = 3


class DataLoader:

    def __init__(self, base_config_path='.', use_drop_subject_list=True, subject_handle=SubjectHandle.INDEPENDENT_DAYS):
        """Data loader

        Helper class for loading different databases from HardDrive.

        Parameters
        ----------
        base_config_path : str
            Path for bci_system.cfg
        use_drop_subject_list : bool
            Whether to use drop subject list from config file or not?
        subject_handle : SubjectHandle
            Type of subject data loading.
            - INDEPENDENT_DAYS: Handle each experiment as an individual subject.
            - MIX_EXPERIMENTS: Train on all experiments of a given subject.
            - BCI_COMP: BCI competition setup, train and test sets are given.
        """
        self._base_dir = Path(init_base_config(base_config_path))
        self.info = mne.Info()

        self._data_path = Path()
        self._db_type = None  # Physionet / TTK / ect...

        self._subject_list = None
        self._drop_subject = set() if use_drop_subject_list else None
        self.subject_handle = subject_handle

    def _reset_db(self):
        self._subject_list = None

    def _validate_db_type(self):
        assert self._db_type is not None, 'Database is not defined.'

    def use_db(self, db_name, config_ver=-1):
        if db_name == EEG_Databases.PHYSIONET:
            self.use_physionet(config_ver)
        elif db_name == EEG_Databases.PILOT_PAR_A:
            self.use_pilot_par_a(config_ver)
        elif db_name == EEG_Databases.PILOT_PAR_B:
            self.use_pilot_par_b(config_ver)
        elif db_name == EEG_Databases.TTK:
            self.use_ttk_db(config_ver)
        elif db_name == EEG_Databases.GAME:
            self.use_game_data(config_ver)
        elif db_name == EEG_Databases.GAME_PAR_C:
            self.use_game_par_c(config_ver)
        elif db_name == EEG_Databases.GAME_PAR_D:
            self.use_game_par_d(config_ver)
        elif db_name == EEG_Databases.BCI_COMP_IV_1:
            self.use_bci_comp_4_1(config_ver)
        elif db_name == EEG_Databases.BCI_COMP_IV_2A:
            self.use_bci_comp_4_2a(config_ver)
        elif db_name == EEG_Databases.BCI_COMP_IV_2B:
            self.use_bci_comp_4_2b(config_ver)
        elif db_name == EEG_Databases.ParadigmC:
            self.use_par_c(config_ver)
        elif db_name == EEG_Databases.EMOTIV_PAR_C:
            self.use_emotiv(config_ver)
        elif db_name == EEG_Databases.GIGA:
            self.use_giga(config_ver)

        else:
            raise NotImplementedError('Database processor for {} db is not implemented'.format(db_name))
        return self

    def _use_db(self, db_type):
        """Loads a specified database."""
        self._data_path = self._base_dir.joinpath(db_type.DIR)
        assert self._data_path.exists(), "Path {} does not exists.".format(self._data_path)
        self._db_type = db_type
        self._reset_db()

        if self._drop_subject is not None:
            self._drop_subject = set(db_type.DROP_SUBJECTS)
        else:
            self._drop_subject = set()

    def get_data_path(self):
        self._validate_db_type()
        return self._data_path

    def use_physionet(self, config_ver=-1):
        self._use_db(Physionet(config_ver))
        return self

    def use_pilot_par_a(self, config_ver=-1):
        self._use_db(PilotDB_ParadigmA(config_ver))
        return self

    def use_pilot_par_b(self, config_ver=-1):
        self._use_db(PilotDB_ParadigmB(config_ver))
        return self

    def use_ttk_db(self, config_ver=-1):
        self._use_db(TTK_DB(config_ver))
        return self

    def use_game_data(self, config_ver=-1):
        self._use_db(GameDB(config_ver))
        return self

    def use_game_par_c(self, config_ver=-1):
        self._use_db(Game_ParadigmC(config_ver))
        return self

    def use_game_par_d(self, config_ver=-1):
        self._use_db(Game_ParadigmD(config_ver))
        return self

    def use_bci_comp_4_1(self, config_ver=-1):
        self._use_db(BciCompIV1(config_ver))
        return self

    def use_bci_comp_4_2a(self, config_ver=-1):
        self._use_db(BciCompIV2a(config_ver))
        return self

    def use_bci_comp_4_2b(self, config_ver=-1):
        self._use_db(BciCompIV2b(config_ver))
        return self

    def use_par_c(self, config_ver=-1):
        self._use_db(ParadigmC(config_ver))
        return self

    def use_emotiv(self, config_ver=-1):
        self._use_db(EmotivParC(config_ver))
        return self

    def use_giga(self, config_ver=-1):
        self._use_db(Giga(config_ver))
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
        self._validate_db_type()
        if type(self._db_type) is Physionet and not self._db_type.CONFIG_VER == 1:
            if REST not in self._db_type.TASK_TO_REC:
                raise ValueError(f'Can not make binary classification. Check values of '
                                 f'TASK_TO_REC in class {type(self._db_type).__name__} '
                                 f'in config.py file. REST should not be commented out!')
        elif REST not in self._db_type.TRIGGER_TASK_CONVERTER:
            implemented_dbs = [GameDB, Game_ParadigmC, ParadigmC, PilotDB_ParadigmA, PilotDB_ParadigmB,
                               TTK_DB, Physionet]
            not_implemented_dbs = [BciCompIV1, BciCompIV2a, BciCompIV2b, Giga]
            if type(self._db_type) in implemented_dbs:
                raise ValueError(f'Can not make binary classification. Check values of '
                                 f'TRIGGER_TASK_CONVERTER in class {type(self._db_type).__name__} '
                                 f'in config.py file. REST should not be commented out!')
            elif type(self._db_type) in not_implemented_dbs:
                raise NotImplementedError(f'{type(self._db_type).__name__} is not implemented for '
                                          f'make_binary_classification.\nYou can comment out some values '
                                          f'from the TRIGGER_TASK_CONVERTER in class {type(self._db_type).__name__} '
                                          f'in config.py file to make the classification binary.')
            elif type(self._db_type) is Game_ParadigmD:
                pass  # always OK
            else:
                raise NotImplementedError(f'class {type(self._db_type).__name__} is not yet integrated...')

    def is_subject_in_drop_list(self, subject):
        self._validate_db_type()
        return subject in self._drop_subject

    def _convert_task(self, record_number=None):
        self._validate_db_type()
        if record_number is None:
            return self._db_type.TRIGGER_TASK_CONVERTER
        return self._db_type.TRIGGER_CONV_REC_TO_TASK.get(record_number)

    def get_command_converter(self):
        self._validate_db_type()
        attr = 'COMMAND_CONV'
        assert hasattr(self._db_type, attr), '{} has no {} attribute'.format(type(self._db_type).__name__, attr)
        return self._db_type.COMMAND_CONV

    def _get_exp_num(self):
        if type(self._db_type) in [Physionet, BciCompIV1, BciCompIV2a, BciCompIV2b, Giga]:
            exp_num = self._db_type.SUBJECT_NUM
        elif type(self._db_type) in [TTK_DB, PilotDB_ParadigmA, PilotDB_ParadigmB, Game_ParadigmC, Game_ParadigmD,
                                     ParadigmC, EmotivParC, GameDB]:
            if type(self._db_type) is EmotivParC:
                file = '*run-001_eeg.xdf'
            elif hasattr(self._db_type, 'CONFIG_VER') and self._db_type.CONFIG_VER >= 1:
                file = '*R01_raw.fif'
            else:
                file = 'rec01.vhdr'
            exp_num = len(sorted(Path(self._data_path).rglob(file)))

            if exp_num == 0:
                if self._db_type.CONFIG_VER >= 1:
                    raise ValueError(f'No files were found for {type(self._db_type).__name__} with config ver '
                                     f'{self._db_type.CONFIG_VER}.\nPlease Download the latest database!')
                else:
                    raise ValueError(f'No files were found for {type(self._db_type).__name__} with config ver '
                                     f'{self._db_type.CONFIG_VER}.\nPlease Download the older database or '
                                     f'change the config number to the latest (-1)!')
        else:
            raise NotImplementedError('get_subject_num is undefined for {}'.format(type(self._db_type).__name__))
        return exp_num

    def get_subject_num(self):
        """Returns the number of available subjects in Database"""
        self._validate_db_type()
        if self.subject_handle is SubjectHandle.INDEPENDENT_DAYS:
            subject_num = self._get_exp_num()

        elif self.subject_handle is SubjectHandle.MIX_EXPERIMENTS:
            if hasattr(self._db_type, 'CONFIG_VER') and self._db_type.CONFIG_VER > 1:
                if type(self._db_type) in [BciCompIV2a, TTK_DB, Physionet, BciCompIV2b, PilotDB_ParadigmA,
                                           PilotDB_ParadigmB, Game_ParadigmC, Game_ParadigmD, BciCompIV1,
                                           Giga]:
                    exp_num = self._get_exp_num()
                    exp_to_subj = self._db_type.SUBJECT_EXP
                    # handling growing db problems: TTK, Par_C, ect...
                    # todo: warning for downloading latest db if subject is missing...
                    subj_list = [subj for subj, exp in exp_to_subj.items() if exp[-1] <= exp_num]
                    subject_num = len(subj_list)
                else:
                    raise NotImplementedError(f'{SubjectHandle.MIX_EXPERIMENTS} is not implemented '
                                              f'for {type(self._db_type).__name__}')
            else:
                raise NotImplementedError(f'{SubjectHandle.MIX_EXPERIMENTS} option only implemented for'
                                          f'CONFIG_VER > 1 .')

        elif self.subject_handle is SubjectHandle.BCI_COMP:
            if hasattr(self._db_type, 'CONFIG_VER') and self._db_type.CONFIG_VER > 1:
                bci_comp = [BciCompIV2a, BciCompIV2b, Giga]
                if type(self._db_type) in bci_comp:
                    subject_num = len(self._db_type.SUBJECT_EXP)
                else:
                    raise ValueError(f'{SubjectHandle.BCI_COMP} is only implemented '
                                     f'for {bci_comp}')
            else:
                raise NotImplementedError(f'{SubjectHandle.BCI_COMP} option only implemented for '
                                          f'CONFIG_VER > 1.')
        else:
            raise NotImplementedError(f'{self.subject_handle} is not implemented.')

        return subject_num

    def get_subject_list(self):
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

    def _generate_filenames_for_subject(self, subject, subject_format_str, runs=1, run_format_str=None):
        if type(runs) is not list:
            runs = [runs]

        for i in runs:
            f = str(self._data_path.joinpath(self._db_type.FILE_PATH))
            f = f.replace('{subj}', subject_format_str.format(subject))
            if run_format_str is not None:
                f = f.replace('{rec}', run_format_str.format(i))
            yield f

    def _generate_physionet_filenames(self, subject, runs):
        return self._generate_filenames_for_subject(subject, '{:03d}', runs, '{:02d}')

    def _generate_pilot_filename(self, subject, runs=1):
        return self._generate_filenames_for_subject(subject, '{}', runs, '{:02d}')

    def _generate_ttk_filename(self, subject, runs=1):
        return self._generate_filenames_for_subject(subject, '{:02d}', runs, '{:02d}')

    def _generate_bci_comp_4_2a_filename(self, subject):
        return self._generate_filenames_for_subject(subject, '{:02d}')

    def _generate_epocplus_filenames(self, subject, runs=1):
        return self._generate_filenames_for_subject(subject, '{:03d}', runs, '{:03d}')

    def _legacy_filename_gen(self, subj):
        if isinstance(self._db_type, TTK_DB):
            fn_gen = self._generate_ttk_filename(subj)
        elif isinstance(self._db_type, (PilotDB_ParadigmA, PilotDB_ParadigmB, GameDB,
                                        Game_ParadigmC, Game_ParadigmD, ParadigmC)):
            fn_gen = self._generate_pilot_filename(subj)
        # elif type(self._db_type) in [BciCompIV1, BciCompIV2a]:
        #     fn_gen = self._generate_ttk_filename(subj)
        # elif type(self._db_type) is BciCompIV2b:
        #     s_ind = subj - 1
        #     s = s_ind // 3 + 1
        #     rec = s_ind % 3 + 1
        #     fn_gen = self._generate_ttk_filename(s, rec)
        elif isinstance(self._db_type, EmotivParC):
            fn_gen = self._generate_epocplus_filenames(subj)
        else:
            raise NotImplementedError(
                'Filename generation for {} with CONFIG_VER=0 is not implemented'.format(type(self._db_type).__name__))
        return fn_gen

    def _get_subj_pattern(self, subj):
        pattern = self._db_type.FILE_PATH
        pattern = pattern.replace('{subj}', '{:03d}'.format(subj))
        pattern = pattern.replace('{rec}', '*')
        return pattern

    def get_filenames_for_subject(self, subj, train=True, train_mask=(False, False, True)):
        """Generating filenames for a defined subject in a database.

        Parameters
        ----------
        subj : int
            Subject number in database.
        train : bool
            Only used in case of SubjectHandle.BCI_COMP. Select train or test set.
        train_mask : tuple of bool, list of bool
            Only used in case of SubjectHandle.BCI_COMP, with BciCompIV2b database.
            Mask for train file selection.

        Returns
        -------
        file_names : list of str
            List containing all of the files corresponding to the subject number.
        """
        subj_num = self.get_subject_num()
        assert subj <= subj_num, f'Subject{subj} is out of subject range. Last subject in db is {subj_num}.' \
                                 f'\nYou may would like to download the latest database.'

        if self.subject_handle is SubjectHandle.INDEPENDENT_DAYS:
            if hasattr(self._db_type, 'CONFIG_VER') and self._db_type.CONFIG_VER >= 1:
                if type(self._db_type) in [Physionet, TTK_DB, PilotDB_ParadigmA, PilotDB_ParadigmB,
                                           Game_ParadigmC, Game_ParadigmD, BciCompIV2a, BciCompIV2b,
                                           BciCompIV1, Giga]:
                    file_names = sorted(self._data_path.rglob(self._get_subj_pattern(subj)))
                    assert len(file_names) > 0, f'No files were found. Try to set CONFIG_VER=0 ' \
                                                f'for {type(self._db_type).__name__} or download the latest database.'
                else:
                    raise NotImplementedError('Filename generation for {} with CONFIG_VER>=1 '
                                              'is not implemented.'.format(type(self._db_type).__name__))
            else:
                file_names = list(self._legacy_filename_gen(subj))

        elif self.subject_handle is SubjectHandle.MIX_EXPERIMENTS:
            if hasattr(self._db_type, 'SUBJECT_EXP'):
                file_names = list()
                for s in self._db_type.SUBJECT_EXP[subj]:
                    file_names.extend(sorted(self._data_path.rglob(self._get_subj_pattern(s))))
            else:
                raise AttributeError(f'{type(self._db_type).__name__} has no attribute called SUBJECT_EXP. '
                                     f'Can not use {SubjectHandle.MIX_EXPERIMENTS} setting.')

        elif self.subject_handle is SubjectHandle.BCI_COMP:
            if hasattr(self._db_type, 'SUBJECT_EXP'):
                if type(self._db_type) in [BciCompIV2a, Giga]:
                    s = self._db_type.SUBJECT_EXP[subj][0 if train else 1]
                    file_names = sorted(self._data_path.rglob(self._get_subj_pattern(s)))
                elif type(self._db_type) is BciCompIV2b:
                    assert len(train_mask) == 3, f'In {BciCompIV2b} there are only 3 train files. ' \
                                                 f'Please define a train_mask with 3 elements.'
                    s = np.asarray(self._db_type.SUBJECT_EXP[subj])
                    mask = list(train_mask) + [False] * 2 if train else [3, 4]
                    file_names = []
                    for i in s[mask]:
                        file_names.extend(sorted(self._data_path.rglob(self._get_subj_pattern(i))))
                else:
                    raise NotImplementedError(f'{type(self._db_type).__name__} is not implemented with '
                                              f'Can not use {SubjectHandle.BCI_COMP} setting.')
            else:
                raise AttributeError(f'{type(self._db_type).__name__} has no attribute called SUBJECT_EXP. '
                                     f'{SubjectHandle.MIX_EXPERIMENTS} setting.')

        else:
            raise NotImplementedError(f'{self.subject_handle} is not implemented.')
        assert len(file_names) > 0, f'No files were found with {self._db_type.FILE_PATH} pattern ' \
                                    f'in {self._data_path} path.'
        return file_names

    def get_task_dict(self):
        self._validate_db_type()
        if type(self._db_type) is Physionet and self._db_type.CONFIG_VER < 1:
            raise NotImplementedError('This method is not implemented for old Physionet config.')
        return self._db_type.TRIGGER_TASK_CONVERTER

    def get_event_id(self):
        self._validate_db_type()
        if type(self._db_type) is Physionet and self._db_type.CONFIG_VER < 1:
            raise NotImplementedError('This method is not implemented for old Physionet config.')
        return self._db_type.TRIGGER_EVENT_ID

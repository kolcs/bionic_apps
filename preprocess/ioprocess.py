import numpy as np
import pickle
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
        NotImplementedError('{} file reading is not implemented.'.format(ext))

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


def generate_filenames(file_path, subject, runs):
    """Filename generator

    Generating filenames from a given string with {subj} and {rec} which are will be replaced.

    Parameters
    ----------
    file_path : str
        special string with {subj} and {rec} substrings
    subject : int
        subject number
    runs : list of int
        list of sessions

    Returns
    -------
    rec : list of str
        filenames for a subject with given runs

    """
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


def _init_interp(interp, epochs, ch_type='eeg'):
    """spatial data creator initializer

    This function initialize the interpreter which can be used to generate spatially distributed data
    from eeg signal if it is None. Otherwise returns it

    Parameters
    ----------
    interp : None | mne.viz.topomap._GridData
        the interpreter
    epochs : mne.Epoch
    ch_type : str

    Returns
    -------
    interp : mne.viz.topomap._GridData
        interpreter for spatially distributed data creation

    """
    if interp is None:
        from mne.channels import _get_ch_type
        from mne.viz.topomap import _prepare_topo_plot

        layout = mne.channels.read_layout('EEG1005')
        ch_type = _get_ch_type(epochs, ch_type)
        picks, pos, merge_grads, names, ch_type = _prepare_topo_plot(
            epochs, ch_type, layout)
        data = epochs.get_data()[0, :, 0]
        im, _, interp = mne.viz.plot_topomap(data, pos, show=False)
    return interp


def _calculate_spatial_data(interp, epochs, crop=True):
    """Spatial data from epochs

    Creates spatially distributed data for each epoch.

    Parameters
    ----------
    interp : mne.viz.topomap._GridData
        interpreter for spatially distributed data creation. Should be initialized first!
    epochs : mne.Epoch
        Data for spatially distributed data generation. Each epoch will have its own spatially data.
        The time points are averaged for each eeg channel.
    crop : bool
        To corp the values to 0 if its out of the eeg circle.

    Returns
    -------
    spatial_list : list of ndarray
        Data

    """
    spatial_list = list()
    data3d = epochs.get_data()

    for i in range(np.size(epochs, 0)):
        ep = data3d[i, :, :]
        ep = np.average(ep, axis=1)  # average time for each channel
        interp.set_values(ep)
        spatial_data = interp()

        # removing data from the border - ROUND electrode system
        if crop:
            r = np.size(spatial_data, axis=0) / 2
            for i in range(int(2 * r)):
                for j in range(int(2 * r)):
                    if np.power(i - r, 2) + np.power(j - r, 2) > np.power(r, 2):
                        spatial_data[i, j] = 0

        spatial_list.append(spatial_data)

    return spatial_list


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


class OfflineDataPreprocessor:
    """
    Preprocessor for edf files. Creates a database, which has all the required information about the eeg files.
    TODO: check what can be removed!!!
    """

    def __init__(self, base_dir, epoch_tmin=0, epoch_tmax=3, use_drop_subject_list=True, window_length=0.5,
                 window_step=0.25, fast_load=True):
        self._base_dir = base_dir
        self._data_path = None
        self._db_type = None  # Physionet / TTK

        self._epoch_tmin = epoch_tmin
        self._epoch_tmax = epoch_tmax  # seconds
        # self._preload = preload
        self._window_length = window_length  # seconds
        self._window_step = window_step  # seconds

        self._data_set = dict()
        self._fast_load = fast_load

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

    def get_subjects(self):
        return list(self._data_set.keys())

    @staticmethod
    def _load_data_from_source(source_files):
        data = dict()
        for filename in source_files:
            d = load_pickle_data(filename)
            data.update(d)

        return data

    def _create_physionet_db(self, db_path, db_source):

        keys = self._db_type.TASK_TO_REC.keys()
        interp = None
        db_filenames = list()

        for s in range(self._db_type.SUBJECT_NUM):
            subj = s + 1

            if subj in self._drop_subject:
                print('Dropping subject {}'.format(subj))
                continue

            subject_data = list()

            for task in keys:
                filenames = generate_filenames(self._data_path + self._db_type.FILE_PATH, subj,
                                               self.convert_rask_to_recs(task))
                raws = [open_raw_file(file, preload=False) for file in filenames]
                raw = raws.pop(0)
                for r in raws:
                    raw.append(r)
                del raws
                raw.rename_channels(lambda x: x.strip('.'))

                # todo: make filtering here...
                # todo: create multi layered picture -- rgb like -> channels
                # todo: for svm use only channels, do not create pictures! SVM oly see vectors...
                # todo: create function parameter, where data type (vector, picture) can be selected!

                rec_num = get_record_number(filenames[0])
                task_dict = self.convert_task(rec_num)

                events, _ = mne.events_from_annotations(raw)
                epochs = mne.Epochs(raw, events, event_id=task_dict, tmin=self._epoch_tmin, tmax=self._epoch_tmax,
                                    preload=False)
                epochs = epochs[task]
                interp = _init_interp(interp, epochs)

                win_epochs = []
                win_num = int((self._epoch_tmax - self._epoch_tmin - self._window_length) / self._window_step)
                for i in range(win_num):
                    ep = epochs.copy().load_data()
                    ep.crop(i * self._window_step, self._window_length + i * self._window_step)
                    data = _calculate_spatial_data(interp, ep)
                    data = [(d, task) for d in data]
                    win_epochs.extend(data)

                subject_data.extend(win_epochs)

            self._data_set[subj] = subject_data

            db_file = '{}subject{}.data'.format(db_path, subj)
            save_pickle_data({subj: subject_data}, db_file)
            db_filenames.append(db_file)
            save_pickle_data(db_filenames, db_path + db_source)

    def _create_db(self):
        # filenames = self._get_filenames()

        if self._db_type is Physionet:
            db_path, db_source = SPATIAL
            db_path = self._base_dir + db_path
            make_dir(db_path)

            data_source = load_pickle_data(db_path + db_source)
            if data_source is not None and self._fast_load:
                self._data_set = self._load_data_from_source(data_source)

            else:
                print('{} file is not found. Creating database.'.format(db_path + db_source))
                self._create_physionet_db(db_path, db_source)

        else:
            raise NotImplementedError('Cannot create subject database for {}'.format(self._db_type))

    def get_split(self, subject, shuffle=True, random_seed=None):
        subjects = list(self._data_set.keys())
        subjects.remove(subject)

        train = list()
        test = self._data_set.get(subject)

        for s in subjects:
            train.extend(self._data_set.get(s))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(train)
            np.random.seed(random_seed)
            np.random.shuffle(test)

        train_x, train_y = zip(*train)
        test_x, test_y = zip(*train)

        return list(train_x), list(train_y), list(test_x), list(test_y)


if __name__ == '__main__':
    # base_dir = "D:/BCI_stuff/databases/"  # MTA TTK
    # base_dir = 'D:/Csabi/'  # Home
    # base_dir = "D:/Users/Csabi/data"  # ITK
    base_dir = "/home/csabi/databases/"  # linux

    proc = OfflineDataPreprocessor(base_dir).use_physionet()
    proc.run()

    # this is how SubjectKFold works:
    subj_k_fold = SubjectKFold(10)
    for train_x, train_y, test_x, test_y in subj_k_fold.split(proc):
        print(train_x[0], train_y[0], test_x[0], test_y[0])

    # todo: continue with svm

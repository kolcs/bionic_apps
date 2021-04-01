from pathlib import Path

import mne
from mne.externals.pymatreader import read_mat

from config import BciCompIV2a, BciCompIV2b
from gui_handler import select_folder_in_explorer
from preprocess import generate_bci_comp_4_2a_filename, init_base_config, generate_ttk_filename


def get_filenames_from_dir(ext):
    ext = ext if ext[0] == '.' else '.' + ext
    path = Path(select_folder_in_explorer('Select base folder which contains the {} files'.format(ext),
                                          'Select directory'))
    filenames = list(path.rglob('*{}'.format(ext)))
    return filenames


def convert_bcicompIV2a():
    base_dir = Path(init_base_config('..'))
    db = BciCompIV2a
    for i in range(db.SUBJECT_NUM):
        subj = i + 1
        filename = next(
            generate_bci_comp_4_2a_filename(str(base_dir.joinpath(db.DIR, db.FILE_PATH)), subj))
        raw = mne.io.read_raw(filename, preload=True, eog=(-3, -2, -1))  # eog channel index

        # correcting eeg channel names
        raw.rename_channels(lambda x: x.strip('EEG-'))
        new_map = {str(i): ch for i, ch in enumerate(['Fc3', 'Fc1', 'FCz', 'Fc2', 'Fc4', 'C5', 'C1', 'C2', 'C6',
                                                      'Cp3', 'Cp1', 'CPz', 'Cp2', 'Cp4', 'P1', 'P2', 'POz'])}
        raw.rename_channels(new_map)
        filename = filename.strip('.gdf') + '_raw.fif'
        raw.save(filename, overwrite=True)


def convert_bcicompIV2b():
    base_dir = Path(init_base_config('..'))
    db = BciCompIV2b
    for subj in range(db.SUBJECT_NUM):
        # file indexing: one subject has many records in different time-period
        s = subj // 3 + 1
        rec = subj % 3 + 1
        filename = next(
            generate_ttk_filename(str(base_dir.joinpath(db.DIR, db.FILE_PATH)), s, rec))
        raw = mne.io.read_raw(filename, preload=True, eog=(-3, -2, -1))  # eog channel index
        raw.rename_channels(lambda x: x.strip('EEG:'))
        filename = filename.strip('.gdf') + '_raw.fif'
        raw.save(filename, overwrite=True)


def _create_raw(eeg, ch_names, ch_types, fs, onset, duration, description):
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=fs)
    raw = mne.io.RawArray(eeg, info)
    annotation = mne.Annotations(onset, duration, description)
    raw = raw.set_annotations(annotation)
    return raw


def convert_bcicompIV1():
    filenames = get_filenames_from_dir('.mat')
    for i, file in enumerate(filenames):
        mat = read_mat(str(file))

        eeg = mat['cnt'].transpose().astype('double') * 1e-7  # convert to 1 Volt unit
        ch_names = mat['nfo']['clab']
        ch_types = ['eeg'] * len(ch_names)
        fs = mat['nfo']['fs']
        classes = mat['nfo']['classes']
        onset = mat['mrk']['pos'] / fs
        duration = [4] * len(onset)
        description = mat['mrk']['y']

        description = description.astype(object)
        description[description == 1] = classes[0]
        description[description == -1] = classes[1]
        filename = file.parent.joinpath('calib_ds_subj{:02d}_raw.fif'.format(i + 1))

        raw = _create_raw(eeg, ch_names, ch_types, fs, onset, duration, description)
        raw.save(str(filename), overwrite=True)


def convert_giga():
    filenames = get_filenames_from_dir('.mat')
    for file in filenames:
        mat = read_mat(str(file))
        raw_list = list()
        for state in ['EEG_MI_train', 'EEG_MI_test']:
            data_dict = mat[state]
            eeg = data_dict['x'].transpose() * 1e-6  # todo: waiting for info! convert to 1 Volt unit
            ch_names = data_dict['chan']
            ch_types = ['eeg'] * len(ch_names)
            fs = data_dict['fs']
            onset = data_dict['t'] / fs
            duration = [4] * len(onset)
            description = data_dict['y_class']

            raw_list.append(_create_raw(eeg, ch_names, ch_types, fs, onset, duration, description))

        raw = mne.concatenate_raws(raw_list)  # todo: same experiment?

        filename = str(file).strip('.mat') + '_raw.fif'
        # filename = file.parent.joinpath('calib_ds_subj{:02d}_raw.fif'.format(i + 1))
        raw.save(str(filename), overwrite=True)


if __name__ == '__main__':
    convert_giga()

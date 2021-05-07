from pathlib import Path

import mne
import numpy as np
from matplotlib import pyplot as plt
from mne.externals.pymatreader import read_mat

from config import BciCompIV2a, BciCompIV2b, Physionet, IMAGINED_MOVEMENT, BOTH_LEGS
from gui_handler import select_folder_in_explorer
from preprocess import DataLoader, init_base_config, \
    generate_ttk_filename, generate_physionet_filenames, generate_bci_comp_4_2a_filename


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


def convert_physionet():
    base_dir = Path(init_base_config('..')).joinpath(Physionet.DIR)
    for s in range(Physionet.SUBJECT_NUM):
        subj = s + 1
        rec_nums = Physionet.TYPE_TO_REC[IMAGINED_MOVEMENT]
        raw_list = list()
        new_rec_num = 1
        for rec in rec_nums:
            filename = next(
                generate_physionet_filenames(str(base_dir.joinpath(Physionet.FILE_PATH)), subj, rec)
            )
            trigger_id = Physionet.TRIGGER_CONV_REC_TO_TASK[rec]
            raw = mne.io.read_raw(filename, preload=True)
            if BOTH_LEGS in trigger_id:
                description = list()
                for t in raw.annotations.description:
                    if t == 'T0':
                        description.append(t)
                    elif t == 'T1':
                        description.append('T3')
                    elif t == 'T2':
                        description.append('T4')
                    else:
                        raise ValueError(f'{t} is not supported...')
                raw.annotations.description = description
            raw_list.append(raw)

            if len(raw_list) == 2:
                raw = mne.io.concatenate_raws(raw_list)
                raw_list = list()
                file = base_dir.joinpath('S{:03d}'.format(subj), 'S{:03d}R{:02d}_raw.fif'.format(subj, new_rec_num))
                file.parent.mkdir(parents=True, exist_ok=True)
                new_rec_num += 1
                raw.save(str(file), overwrite=True)


def _save_sessions(subj, raw, start_mask, end_mask, path, session_num=13, plot=False):
    start_ind = np.arange(len(start_mask))[start_mask]
    end_ind = np.arange(len(end_mask))[end_mask]
    assert len(start_ind) == session_num, f'Incorrect start Triggers at subject {subj}'
    assert len(end_ind) == session_num, f'Incorrect end Triggers at subject {subj}'

    # remove first 3 True from start mask -- eye, train1, train2 sessions
    start_mask[start_ind[:3]] = False
    end_mask[end_ind[:3]] = False

    tmins = raw.annotations.onset[start_mask]
    tmaxs = raw.annotations.onset[end_mask]
    for i, tmin in enumerate(tmins):
        tmax = tmaxs[i]
        sess = raw.copy()
        sess.crop(tmin, tmax + 1)
        file = path.joinpath('S{:03d}'.format(subj), 'S{:03d}R{:02d}_raw.fif'.format(subj, i + 1))
        file.parent.mkdir(parents=True, exist_ok=True)
        sess.save(str(file), overwrite=True)

        if plot:
            sess.plot(block=False)

    if plot:
        raw.plot(block=False)
        plt.show()


def convert_ttk():
    base_dir = Path(init_base_config('..'))
    proc = DataLoader(str(base_dir)).use_ttk_db()
    assert not proc._db_type.USE_NEW_CONFIG, 'File conversion only avaliable for old config setup.'

    for subj in proc.get_subject_list():
        filename = next(generate_ttk_filename(proc.get_file_path_for_db_filename_gen(), subj))
        raw = mne.io.read_raw(filename, preload=True)

        start_mask = (raw.annotations.description == 'Response/R  1') | (raw.annotations.description == 'Stimulus/S 16')
        end_mask = raw.annotations.description == 'Stimulus/S 12'
        start_ind = np.arange(len(start_mask))[start_mask]
        end_ind = np.arange(len(end_mask))[end_mask]
        trigger_num = 13

        if subj == 2:
            # Session end trigger is missing, creating end mask from start_mask
            end_mask = np.array([False] * len(start_mask))
            ind = start_ind[1:] - 1
            end_mask[ind] = True
            end_mask[-1] = True
        elif subj == 12:
            # One extra start trigger at the end of the record...
            start_mask[start_ind[-1]] = False
        elif subj == 13:
            # 12 sessions instead of 10
            trigger_num += 2
        elif subj == 18:
            start_mask[start_ind[:2]] = False  # wrong trigger
            start_mask[start_ind[-2]] = False  # wrong trigger
            start_mask[start_ind[-3]] = False  # wrong session, with no end...
            end_mask[end_ind[0]] = False

        _save_sessions(subj, raw, start_mask, end_mask, proc.get_data_path(), trigger_num)


def convert_bad_ttk():
    base_dir = Path(init_base_config('..'))
    proc = DataLoader(str(base_dir)).use_ttk_db()
    assert not proc._db_type.USE_NEW_CONFIG, 'File conversion only avaliable for old config setup.'

    for subj in [1, 9, 17]:
        if subj == 1:
            filenames = generate_ttk_filename(proc.get_file_path_for_db_filename_gen(), subj, [1, 2, 3])
        elif subj == 9:
            # rec02.vhdr file is corrupted...
            filenames = generate_ttk_filename(proc.get_file_path_for_db_filename_gen(), subj, [1])
        else:
            filenames = generate_ttk_filename(proc.get_file_path_for_db_filename_gen(), subj, [1, 2])

        raw = mne.io.concatenate_raws([mne.io.read_raw(file) for file in filenames])

        start_mask = (raw.annotations.description == 'Response/R  1') | (raw.annotations.description == 'Stimulus/S 16')
        end_mask = raw.annotations.description == 'Stimulus/S 12'
        start_ind = np.arange(len(start_mask))[start_mask]
        trigger_num = 13

        if subj == 1:
            start_mask[start_ind[8]] = False  # wrong start...
            start_ind = np.arange(len(start_mask))[start_mask]
            # Session end trigger is missing, creating end mask from start_mask
            end_mask = np.array([False] * len(start_mask))
            ind = start_ind[1:] - 1
            end_mask[ind] = True
            end_mask[-1] = True
        elif subj == 9:
            start_mask[start_ind[-1]] = False  # wrong start...
            trigger_num = 9
        elif subj == 17:
            end_ind = np.arange(len(end_mask))[end_mask]
            start_mask[start_ind[5]] = False  # wrong session
            end_mask[end_ind[5]] = False  # wrong session
            trigger_num = 12

        _save_sessions(subj, raw, start_mask, end_mask, proc.get_data_path(), trigger_num)


def convert_all_ttk():
    convert_ttk()
    convert_bad_ttk()


if __name__ == '__main__':
    convert_all_ttk()

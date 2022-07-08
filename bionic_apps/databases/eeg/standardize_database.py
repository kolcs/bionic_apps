from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np
from matplotlib import pyplot as plt
from pymatreader import read_mat

from bionic_apps.databases import EEG_Databases, get_eeg_db_name_by_filename
from bionic_apps.databases.eeg.defaults import IMAGINED_MOVEMENT, BOTH_LEGS
from bionic_apps.handlers.gui import select_folder_in_explorer
from bionic_apps.preprocess.io import DataLoader, get_epochs_from_raw

BASE_PATH = '.'


def get_filenames_from_dir(ext):
    ext = ext if ext[0] == '.' else '.' + ext
    path = Path(select_folder_in_explorer('Select base folder which contains the {} files'.format(ext),
                                          'Select directory'))
    filenames = list(path.rglob('*{}'.format(ext)))
    return filenames


def _check_annotations(raws):
    file = Path(raws[0].filenames[0]).as_posix()
    loader = DataLoader('.').use_db(get_eeg_db_name_by_filename(file))
    prev_annot_num = 0
    for raw in raws:
        annot_num = len(raw.annotations.description)
        assert annot_num > 0, 'Annotations are missing...'
        # if prev_annot_num != 0 and prev_annot_num != annot_num:
        #     print(f'Number of annotations are not equal: {prev_annot_num} != {annot_num}')
        prev_annot_num = annot_num

    raw = mne.io.concatenate_raws(raws)

    # if no error OK.
    ep = get_epochs_from_raw(raw, loader.get_task_dict(), event_id=loader.get_event_id())
    # raw.plot()
    # ep.plot(block=True)


def _save_sessions(subj, raw, start_mask, end_mask, path, session_num=13, drop_first=3,
                   before_session=0, after_session=1., after_end=1., sess_num=0,
                   plot=False, check_saved=False):
    start_ind = np.arange(len(start_mask))[start_mask]
    end_ind = np.arange(len(end_mask))[end_mask]
    assert len(start_ind) == session_num, f'Incorrect start Triggers at subject {subj}'
    assert len(end_ind) == session_num, f'Incorrect end Triggers at subject {subj}'

    # remove first 3 True from start mask -- eye, train1, train2 sessions
    if drop_first != 0:
        start_mask[start_ind[:drop_first]] = False
        end_mask[end_ind[:drop_first]] = False

    saved_sess = []
    tmins = raw.annotations.onset[start_mask]
    tmaxs = raw.annotations.onset[end_mask]
    for i, tmin in enumerate(tmins):
        tmax = tmaxs[i]
        sess = raw.copy()
        if i < len(tmins) - 1:
            sess.crop(tmin - before_session, tmax + after_session)
        else:
            sess.crop(tmin - before_session, tmax + after_end)

        file = path.joinpath('S{:03d}'.format(subj), 'S{:03d}R{:02d}_raw.fif'.format(subj, sess_num + i + 1))
        file.parent.mkdir(parents=True, exist_ok=True)
        sess.save(str(file), overwrite=True)
        saved_sess.append(mne.io.read_raw_fif(str(file)))

        if plot:
            sess.plot(block=False)
        if not (plot and not check_saved or not check_saved):
            saved_sess[i].plot(block=(not plot and i == len(tmins) - 1))

    if plot:
        raw.plot(block=True)
        plt.show()

    _check_annotations(saved_sess)


def _add_missing_bci_comp_4_2_triggers(filename, raw):
    matfile = str(filename.with_suffix('.mat'))
    mat = read_mat(matfile)
    new_tiggers = mat['classlabel']
    tgdict = {i + 1: str(tg) for i, tg in enumerate([769, 770, 771, 772])}
    new_tiggers = list(map(lambda x: tgdict[x], new_tiggers))
    raw.annotations.description[raw.annotations.description == '783'] = new_tiggers


def convert_bcicompIV2a():
    loader = DataLoader(BASE_PATH).use_bci_comp_4_2a()
    path = loader.get_data_path()
    files = sorted(path.rglob('*{}'.format('.gdf')), key=lambda x: x.stem[-1], reverse=True)
    files = sorted(files, key=lambda x: x.stem[:-1])
    for subj, filename in enumerate(files):
        subj += 1
        raw = mne.io.read_raw(filename, preload=True, eog=(-3, -2, -1))  # eog channel index

        # correcting eeg channel names
        raw.rename_channels(lambda x: x.strip('EEG-'))
        new_map = {str(i): ch for i, ch in enumerate(['Fc3', 'Fc1', 'FCz', 'Fc2', 'Fc4', 'C5', 'C1', 'C2', 'C6',
                                                      'Cp3', 'Cp1', 'CPz', 'Cp2', 'Cp4', 'P1', 'P2', 'POz'])}
        raw.rename_channels(new_map)

        if filename.stem[-1] == 'E':  # adding correct trigger numbers ot evalset
            _add_missing_bci_comp_4_2_triggers(filename, raw)
        elif filename.stem[-1] == 'T':
            pass
        else:
            raise NotImplementedError

        # Session end trigger is missing, creating end mask from start_mask
        start_mask = raw.annotations.description == '32766'
        end_mask = start_mask.copy()
        end_mask[0] = False
        end_mask[-1] = True
        if subj == 7:
            drop_first = 1
        else:
            drop_first = 3
        session_num = 6 + drop_first
        _save_sessions(subj, raw, start_mask, end_mask, path, session_num=session_num, drop_first=drop_first,
                       after_end=5.7)


def convert_bcicompIV2b():
    loader = DataLoader(BASE_PATH, use_drop_subject_list=False).use_bci_comp_4_2b()
    path = loader.get_data_path()
    files = sorted(path.rglob('*{}'.format('.gdf')))
    for i, filename in enumerate(files):
        subj = i + 1
        raw = mne.io.read_raw(filename, preload=True, eog=(-3, -2, -1))  # eog channel index
        raw.rename_channels(lambda x: x.strip('EEG:'))
        if i % 5 < 3:
            pass
        else:
            _add_missing_bci_comp_4_2_triggers(filename, raw)

        # Session end trigger is missing, creating end mask from start_mask
        start_mask = raw.annotations.description == '32766'
        end_mask = start_mask.copy()
        end_mask[0] = False
        end_mask[-1] = True
        print(subj, filename)

        if subj in [2, 24]:
            drop_first = 0
        else:
            drop_first = 1

        if subj in [17, 22]:
            session_num = 7 + drop_first
        elif subj == 9:
            session_num = 3 + drop_first
        elif subj == 36:
            session_num = 8 + drop_first
        elif i % 5 < 2:
            session_num = 6 + drop_first
        else:
            session_num = 4 + drop_first
        _save_sessions(subj, raw, start_mask, end_mask, path, session_num=session_num, drop_first=drop_first,
                       after_end=5.7)


def _create_raw(eeg, ch_names, ch_types, fs, onset, duration, description, rec_date=None):
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=fs)
    raw = mne.io.RawArray(eeg, info)
    annotation = mne.Annotations(onset, duration, description)
    raw = raw.set_annotations(annotation)
    if rec_date is not None:
        raw.set_meas_date(rec_date)
    return raw


def convert_bcicompIV1():
    path = Path(select_folder_in_explorer('Select base folder which contains the BCI comp IV 1 .mat files',
                                          'Select directory'))
    new_subj = 0
    files = sorted(path.rglob('BCICIV_calib_*_1000Hz.mat'))
    assert len(files) > 0, 'No files were found...'

    for subj, file in enumerate(files):
        subj += 1
        # if subj in [3, 4, 5]:
        #     continue  # artificial data
        for k in range(2):
            if k == 1:
                continue  # todo: problem with varying eval set length...
                # file = str(file).replace('calib', 'eval')
            new_subj += 1
            mat = read_mat(str(file))

            eeg = mat['cnt'].transpose().astype('double') * 1e-7  # convert to 1 Volt unit
            ch_names = mat['nfo']['clab']
            ch_types = ['eeg'] * len(ch_names)
            fs = mat['nfo']['fs']
            classes = mat['nfo']['classes']
            date = mat['__header__'].decode('utf-8').split('Created on:')[-1]
            rec_date = datetime.strptime(date, ' %a %b\t%d  %H:%M:%S %Y').replace(tzinfo=timezone.utc)

            if k == 0:
                onset = mat['mrk']['pos'] / fs
                description = mat['mrk']['y']
                duration = [4] * len(onset)
            else:
                continue  # todo: problem with varying eval set length...
                # label_file = file.split('.')
                # label_file = label_file[0] + '_true_y.' + label_file[1]
                # mat = read_mat(label_file)
                # true_y = np.nan_to_num(mat['true_y'])
                # onset_mask = [False] + [(True if true_y[i] != true_y[i - 1] and true_y[i] != 0 else False) for i in
                #                         range(1, len(true_y))]
                # end_mask = [(True if true_y[i] != true_y[i + 1] and true_y[i] != 0 else False) for i in
                #             range(len(true_y) - 1)] + [False]
                # end = np.arange(len(true_y))[end_mask] / fs
                #
                # onset = np.arange(len(true_y))[onset_mask] / fs
                # description = true_y[onset_mask]
                # duration = [t_end - onset[i] for i, t_end in enumerate(end)]

            description = description.astype(object)
            description[description == -1] = classes[0]
            description[description == 1] = classes[1]

            start_mask = [True]
            for i in range(1, len(onset)):
                start = True if onset[i] - onset[i - 1] > 9 else False
                start_mask.append(start)
            end_mask = start_mask.copy()
            end_mask[0] = False
            end_mask[-1] = True

            raw = _create_raw(eeg, ch_names, ch_types, fs, onset, duration, description, rec_date)
            _save_sessions(new_subj, raw, start_mask, end_mask, path, session_num=np.sum(start_mask), drop_first=0,
                           before_session=1, after_session=-22, after_end=5)


def convert_giga():
    path = Path(select_folder_in_explorer('Select base folder which contains the Giga database .mat files',
                                          'Select directory'))
    files = sorted(path.rglob('sess*_subj*_EEG_MI.mat'), key=lambda x: x.stem.split('subj')[0])
    files = sorted(files, key=lambda x: x.stem.split('subj')[-1])
    assert len(files) > 0, 'No files were found...'

    for subj, file in enumerate(files):
        subj += 1
        mat = read_mat(str(file))

        prev_onset = 0
        for state in ['EEG_MI_train', 'EEG_MI_test']:
            data_dict = mat[state]
            eeg = np.round(data_dict['x'].transpose(), 1) * 1e-6
            emg = np.round(data_dict['EMG'].transpose(), 1) * 1e-6
            data = np.concatenate((eeg, emg), axis=0)
            ch_names = data_dict['chan'] + data_dict['EMG_index']
            ch_types = ['eeg'] * len(data_dict['chan']) + ['emg'] * len(data_dict['EMG_index'])
            fs = data_dict['fs']
            onset = data_dict['t'] / fs
            duration = [4] * len(onset)
            description = data_dict['y_class']
            date = mat['__header__'].decode('utf-8').split('Created on:')[-1]
            rec_date = datetime.strptime(date, ' %a %b\t%d  %H:%M:%S %Y').replace(tzinfo=timezone.utc)

            start_mask = [True]
            for i in range(1, len(onset)):
                start = True if onset[i] - onset[i - 1] > 20 else False
                start_mask.append(start)
            end_mask = start_mask.copy()
            end_mask[0] = False
            end_mask[-1] = True

            raw = _create_raw(data, ch_names, ch_types, fs, onset, duration, description, rec_date)
            _save_sessions(subj, raw, start_mask, end_mask, path, session_num=np.sum(start_mask), drop_first=0,
                           before_session=1, after_session=-50, after_end=5, sess_num=prev_onset)
            prev_onset = np.sum(start_mask)


def convert_physionet():
    loader = DataLoader(BASE_PATH, use_drop_subject_list=False).use_physionet(config_ver=0)
    assert loader._db_type.CONFIG_VER == 0, 'File conversion only avaliable for CONFIG_VER=0'
    for s in range(loader._db_type.SUBJECT_NUM):
        subj = s + 1
        rec_nums = loader._db_type.TYPE_TO_REC[IMAGINED_MOVEMENT]
        raw_list = list()
        new_rec_num = 1
        for rec in rec_nums:
            filename = next(
                loader._generate_physionet_filenames(subj, rec)
            )
            trigger_id = loader._db_type.TRIGGER_CONV_REC_TO_TASK[rec]
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
                file = loader.get_data_path().joinpath('S{:03d}'.format(subj),
                                                       'S{:03d}R{:02d}_raw.fif'.format(subj, new_rec_num))
                file.parent.mkdir(parents=True, exist_ok=True)
                new_rec_num += 1
                raw.save(str(file), overwrite=True)


def convert_ttk():
    loader = DataLoader(BASE_PATH, use_drop_subject_list=True).use_ttk_db(config_ver=0)
    assert loader._db_type.CONFIG_VER == 0, 'File conversion only avaliable for CONFIG_VER=0'

    for subj in loader.get_subject_list():
        filename = loader.get_filenames_for_subject(subj)[0]
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

        _save_sessions(subj, raw, start_mask, end_mask, loader.get_data_path(), trigger_num)


def convert_bad_ttk():
    loader = DataLoader(BASE_PATH, use_drop_subject_list=False).use_ttk_db(config_ver=0)
    assert loader._db_type.CONFIG_VER == 0, 'File conversion only avaliable for CONFIG_VER=0'

    for subj in [1, 9, 17]:
        if subj == 1:
            filenames = loader._generate_ttk_filename(subj, [1, 2, 3])
        elif subj == 9:
            # rec02.vhdr file is corrupted...
            filenames = loader._generate_ttk_filename(subj, [1])
        else:
            filenames = loader._generate_ttk_filename(subj, [1, 2])

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

        _save_sessions(subj, raw, start_mask, end_mask, loader.get_data_path(), trigger_num)


def convert_all_ttk():
    convert_ttk()
    convert_bad_ttk()


def convert_pilot_par_a():
    loader = DataLoader(BASE_PATH, use_drop_subject_list=True).use_db(EEG_Databases.PILOT_PAR_A, config_ver=0)
    assert loader._db_type.CONFIG_VER == 0, 'File conversion only avaliable for CONFIG_VER=0'

    for subj in loader.get_subject_list():
        filename = loader.get_filenames_for_subject(subj)[0]
        raw = mne.io.read_raw(filename, preload=True)
        start_mask = (raw.annotations.description == 'Response/R  1') | (
                raw.annotations.description == 'Stimulus/S 16')
        end_mask = raw.annotations.description == 'Stimulus/S 12'
        _save_sessions(subj, raw, start_mask, end_mask, loader.get_data_path())


def convert_pilot_par_b():
    loader = DataLoader(BASE_PATH, use_drop_subject_list=True).use_db(EEG_Databases.PILOT_PAR_B, config_ver=0)
    assert loader._db_type.CONFIG_VER == 0, 'File conversion only avaliable for CONFIG_VER=0'

    for subj in loader.get_subject_list():
        if subj in [7, 9]:
            filenames = list(loader._generate_pilot_filename(subj, [1, 2]))
        else:
            filenames = loader.get_filenames_for_subject(subj)

        raw = mne.io.concatenate_raws([mne.io.read_raw(file, preload=True) for file in filenames])

        start_mask = (raw.annotations.description == 'Response/R  1') | (
                raw.annotations.description == 'Stimulus/S 16')
        end_mask = raw.annotations.description == 'Stimulus/S 12'
        trigger_num = 9
        drop_first = 1

        if subj < 5:
            trigger_num = 13
            drop_first = 3
        elif subj in [5, 6]:
            trigger_num = 11
        elif subj == 9:
            start_ind = np.arange(len(start_mask))[start_mask]
            start_mask[start_ind[3]] = False

        _save_sessions(subj, raw, start_mask, end_mask, loader.get_data_path(), trigger_num, drop_first)


def convert_game_par_c_and_d():
    db_list = [EEG_Databases.GAME_PAR_C, EEG_Databases.GAME_PAR_D]
    for db_name in db_list:
        loader = DataLoader(BASE_PATH, use_drop_subject_list=True).use_db(db_name, config_ver=0)
        assert loader._db_type.CONFIG_VER == 0, 'File conversion only avaliable for CONFIG_VER=0'

        for subj in loader.get_subject_list():
            filename = loader.get_filenames_for_subject(subj)[0]
            raw = mne.io.read_raw(filename, preload=True)
            start_mask = (raw.annotations.description == 'Response/R  1') | (
                    raw.annotations.description == 'Stimulus/S 16')
            end_mask = raw.annotations.description == 'Stimulus/S 12'
            drop_first = 1
            session_num = drop_first + 5
            _save_sessions(subj, raw, start_mask, end_mask, loader.get_data_path(), session_num, drop_first,
                           plot=False)


if __name__ == '__main__':
    convert_giga()

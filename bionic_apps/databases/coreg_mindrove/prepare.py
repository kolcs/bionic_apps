from pathlib import Path

import numpy as np
import pandas as pd

from bionic_apps.databases.eeg.standardize_database import _create_raw

DATA_PATH = Path(r'D:\Users\Csabi\OneDrive - Pázmány Péter Katolikus Egyetem\MindRove project\database\Coreg_data')

EMG_CHS = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6', 'EMG7', 'EMG8']
EEG_CHS = ['EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6']
LABEL_CONVERTER = (
        ['Idle'] +
        (['Rest'] + ['Thumb'] * 3) * 3 +
        (['Rest'] + ['Index'] * 3) * 3 +
        (['Rest'] + ['Middle'] * 3) * 3 +
        (['Rest'] + ['Ring'] * 3) * 3 +
        (['Rest'] + ['Small'] * 3) * 3 +
        (['Rest'] + ['Wrist FW'] * 3) * 3 +
        (['Rest'] + ['Wrist BCK'] * 3) * 3 +
        ['Idle']
)

FS = 500


def get_annotated_mindrove_raw(file, mode='merged', plot=False):
    assert file.suffix == '.csv', f'only .csv files are accepted'
    df = pd.read_csv(file, sep=',', encoding='utf8')
    data = df[EEG_CHS + EMG_CHS] * 1e-6
    task_numbers = df['Task_number'].values

    tr = np.ediff1d(task_numbers)
    tr_start = np.insert(tr, 0, 1)
    tr_end = np.append(tr, 1)
    tr_start = np.arange(len(tr_start))[tr_start > 0]
    tr_end = np.arange(len(tr_end))[tr_end > 0] + 1
    assert len(tr_start) == len(tr_end)

    if mode == 'distinct':
        pass
    elif mode == 'merged':
        tr_start_merged, tr_end_merged = [], []
        for i in range(len(tr_start)):
            if i == 0 or i + 1 == len(tr_start) or i % 4 == 1 or i % 4 == 2:
                tr_start_merged.append(tr_start[i])
            if i == 0 or i + 1 == len(tr_start) or i % 4 == 0 or i % 4 == 1:
                tr_end_merged.append(tr_end[i])
        if len(tr_start_merged) > len(tr_end_merged):
            tr_start_merged.pop(-2)
        elif len(tr_start_merged) < len(tr_end_merged):
            tr_end_merged.pop(-2)
        tr_start, tr_end = np.array(tr_start_merged), np.array(tr_end_merged)
    else:
        raise NotImplementedError(f'Mode {mode} is not implemented.')

    onset = tr_start / FS
    duration = tr_end / FS - onset
    ep_labels = np.array([LABEL_CONVERTER[int(lab)] for lab in task_numbers[tr_start]])

    raw = _create_raw(data.T,
                      ch_names=EEG_CHS + EMG_CHS,
                      ch_types=['eeg'] * len(EEG_CHS) + ['emg'] * len(EMG_CHS),
                      fs=FS, onset=onset, duration=duration,
                      description=ep_labels
                      )

    if plot:
        raw.plot(block=True)

    return raw


def reannotate_mindrove(base_dir=DATA_PATH, ep_mode='merged'):
    assert base_dir.exists(), f'Path {base_dir} is not available.'
    emg_files = sorted(base_dir.rglob('*.csv'))
    for j, file in enumerate(emg_files):
        j += 1
        subj = file.stem.split('_')[-1]
        print(f'Subject{j} - {subj}')

        raw = get_annotated_mindrove_raw(file, mode=ep_mode, plot=False)

        raw_cp = raw.copy()
        iir_params = dict(order=5, ftype='butter', output='sos')
        raw_cp.filter(l_freq=1, h_freq=40, method='iir', iir_params=iir_params, skip_by_annotation='edge',
                      n_jobs=-1, picks='eeg')
        raw_cp.filter(l_freq=20, h_freq=150, method='iir', iir_params=iir_params, skip_by_annotation='edge',
                      n_jobs=-1, picks='emg')

        raw_cp.plot(block=True)

        raw.set_annotations(raw_cp.annotations)
        file = str(base_dir.joinpath(f'subject{j:03d}_raw.fif'))
        raw.save(file)

        # # testing:
        # from mne.io import read_raw_fif
        # sraw = read_raw_fif(file)
        #
        # raw.plot(title='Modified, before save')
        # sraw.plot(block=True, title='After save')


if __name__ == '__main__':
    reannotate_mindrove()

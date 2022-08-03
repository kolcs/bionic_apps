from pathlib import Path

import pandas as pd
import numpy as np

import putemg_download
from bionic_apps.databases.eeg.standardize_database import _create_raw

FS = 5120  # Hz
LABEL_CONVERTER = {
    -1: ['Rest'],
    0: ['Idle'],
    1: ['Fist'],
    2: ['Flexion'],
    3: ['Extension'],
    6: ['pinch thumb-index'],
    7: ['pinch thumb-middle'],
    8: ['pinch thumb-ring'],
    9: ['pinch thumb-small']
}


def get_annotated_raw(file, plot=False):
    df = pd.read_hdf(file, sep=',', encoding='utf8')
    emg_cols = [col for col in df if 'EMG' in col]

    data = df[emg_cols].to_numpy() * 5 / 2 ** 12 * 1000 / 200 * 1e-3
    task_numbers = df['TRAJ_GT'].to_numpy()

    tr = np.ediff1d(task_numbers)
    tr_start = np.insert(tr, 0, 1)
    tr_end = np.append(tr, 1)
    tr_start = np.arange(len(tr_start))[tr_start > 0]
    tr_end = np.arange(len(tr_end))[tr_end > 0] + 1
    assert len(tr_start) == len(tr_end)

    onset = tr_start / FS
    duration = tr_end / FS - onset
    ep_labels = np.array([LABEL_CONVERTER[int(lab)] for lab in task_numbers[tr_start]]).ravel()

    raw = _create_raw(data.T,
                      ch_names=emg_cols,
                      ch_types=['emg'] * len(emg_cols),
                      fs=FS, onset=onset, duration=duration,
                      description=ep_labels
                      )

    if plot:
        raw.plot(block=True)

    return raw


def main():
    base_dir = Path('Data-HDF5')

    if not base_dir.exists():
        from unittest.mock import patch
        import sys
        testargs = [f"{__file__}", 'emg_gestures', 'data-hdf5']
        with patch.object(sys, 'argv', testargs):
            putemg_download.main()

    files = sorted(base_dir.glob('*repeats_long*.hdf5'))
    for j, file in enumerate(files):
        print(f'Progress: {j * 100. / len(files):.2f} %')
        raw = get_annotated_raw(file, plot=False)
        file = str(base_dir.joinpath(f'subject{j:03d}_raw.fif'))
        raw.save(file, overwrite=True)


if __name__ == '__main__':
    main()

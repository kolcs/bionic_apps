from enum import Enum
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder


class HDF5Dataset:

    def __init__(self, filename, feature_params=None):
        if feature_params is None:
            feature_params = {}

        self.filename = Path(filename)
        self.filename.parent.mkdir(parents=True, exist_ok=True)

        self.mode = None
        self.file = None
        self.dset_x = None
        self.y = []
        self.subj_meta = []
        self.ep_meta = []
        self._last_ep_num = 0
        self._fs = None
        self._orig_mask = []

        self.feature_params = feature_params.copy()
        self._validate_feat_params()

    def _validate_feat_params(self):
        validated_f_pars = {}
        for key, val in self.feature_params.items():
            if isinstance(val, Enum):
                validated_f_pars[key] = val.name
            elif isinstance(val, dict):
                for k, v in val.items():
                    if k == 'pipeline':
                        validated_f_pars[k] = type(v).__name__
                    else:
                        validated_f_pars[k] = str(v) if not isinstance(v, (int, float)) else v
            elif isinstance(val, (int, float, str)):
                validated_f_pars[key] = val
            elif isinstance(val, list):
                validated_f_pars[key] = np.array(sorted(val))
            else:
                raise ValueError(f'Can not save meta data with type {type(val)}.')
        self.feature_params = validated_f_pars

    def _open(self, mode):
        self.mode = mode
        self.file = h5py.File(str(self.filename), self.mode, libver='latest')

    def add_data(self, data, label, subj, ep_group, orig_mask, fs):
        if self.mode is None:
            self._open('w')
            self.dset_x = self.file.create_dataset('x', data=data,
                                                   maxshape=(None, *data.shape[1:]),
                                                   compression="lzf",
                                                   chunks=(1, *data.shape[1:]))

        elif self.mode == 'r':
            raise IOError('Can not write file in read mode. Close it first.')

        else:
            next_data = self.dset_x.shape[0]
            self.dset_x.resize((next_data + data.shape[0], *data.shape[1:]))
            self.dset_x[next_data:, ...] = data

        self.y.extend(label)
        self.subj_meta.extend(subj)
        ep_group = np.array(ep_group)
        self.ep_meta.extend(ep_group + self._last_ep_num)
        self._orig_mask.extend(orig_mask)
        self._last_ep_num += np.max(ep_group) + 1
        if self._fs is None:
            self._fs = fs
        else:
            assert fs == self._fs, 'Sampling frequency has changed'

    def _close(self, delete=False):
        if self.mode == 'w':
            y = np.array(self.y)
            if y.dtype.kind in ['U', 'S']:
                y = np.char.encode(y, encoding='utf-8')
            self.file.attrs.create('y', y)
            self.file.attrs.create('fs', self._fs)
            self.file.attrs.create('subject', np.array(self.subj_meta))
            self.file.attrs.create('ep_group', np.array(self.ep_meta))
            self.file.attrs.create('orig_mask', np.array(self._orig_mask))
            for key, val in self.feature_params.items():
                assert isinstance(val, (str, int, float, np.ndarray)), f'Can not save meta data with type {type(val)}.'
                self.file.attrs.create(key, val)

        if self.mode is not None:
            self.mode = None

        if self.file is not None:
            self.file.close()
            self.file = None
            self.dset_x = None

        if delete:
            self.filename.unlink(missing_ok=True)

    def close(self):
        self._close()

    def get_data(self, ind):
        if self.mode is None:
            self._open('r')
        elif self.mode == 'w':
            raise IOError('Can not read from file in write mode. Close it first.')

        if isinstance(ind, int):
            pass
        elif ind.dtype == 'bool':
            ind = np.arange(len(ind))[ind]
        return self.file['x'][ind]

    def _get_meta(self, key='all'):
        if self.mode is None:
            self._open('a')
        if key == 'all':
            return self.file.attrs['subject'], self.file.attrs['ep_group'], \
                   self.file.attrs['y'].astype('U'), self.file.attrs['fs']
        elif key == 'y':
            return self.file.attrs['y'].astype('U')
        else:
            return self.file.attrs[key]

    def get_subject_group(self):
        return self._get_meta('subject')

    def get_epoch_group(self):
        return self._get_meta('ep_group')

    def get_y(self):
        return self._get_meta('y')

    def get_fs(self):
        return self._get_meta('fs')

    def get_orig_mask(self):
        return self._get_meta('orig_mask')

    def exists(self):
        if not self.filename.exists():
            return False
        try:
            self._open('r')
        except OSError:
            self._close(delete=True)
            return False

        for key, val in self.feature_params.items():
            if key not in self.file.attrs:
                self._close(delete=True)
                return False

            if key == 'subjects':
                assert isinstance(val, (str, np.ndarray)), f'type {type(val)} is not accepted for subjects'
                subjects = self.file.attrs[key]
                if isinstance(subjects, str) and subjects == 'all':
                    continue
                elif all([s in subjects for s in val]):
                    continue

            if self.file.attrs[key] != val:
                self._close(delete=True)
                return False

        self._close()
        return True

    def __del__(self):
        self.close()


def init_hdf5_db(db_filename):
    db = HDF5Dataset(db_filename)
    subj_ind, ep_ind, y, fs = db._get_meta()
    le = LabelEncoder()
    y = le.fit_transform(y)
    return db, y, subj_ind, ep_ind, le, fs

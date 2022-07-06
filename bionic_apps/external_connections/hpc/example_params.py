from bionic_apps.ai import ClassifierType
from bionic_apps.databases import EEG_Databases
from bionic_apps.feature_extraction import FeatureType
from bionic_apps.offline_analyses import test_eegdb_within_subject
from bionic_apps.preprocess.io import SubjectHandle
import numpy as np

LOG_DIR = 'eeg_net'
TEST_NAME = 'window_inv'

hpc_submit_script = f'{LOG_DIR}/submit_gpu.sh'

test_func = test_eegdb_within_subject

default_kwargs = dict(
    db_name=EEG_Databases.PHYSIONET,
    feature_type=FeatureType.RAW,
    epoch_tmin=0, epoch_tmax=4,
    window_len=2, window_step=.1,
    feature_kwargs=None,
    use_drop_subject_list=True,
    filter_params=None,
    do_artefact_rejection=True,
    balance_data=True,
    subject_handle=SubjectHandle.INDEPENDENT_DAYS,
    n_splits=5,
    classifier_type=ClassifierType.EEG_NET,
    classifier_kwargs=dict(
        epochs=500,
        patience=15
    ),
    # db_file='tmp/database.hdf5', log_file='out.csv', base_dir='.',
    save_res=True,
    fast_load=True, subjects='all',
    augment_data=False
)

# generating test params here...
_windows = [(win_len, win_step)
            for win_len in np.arange(.5, 4.1, .5)
            for win_step in [0, .01, .05]]

test_kwargs = []

for win_len, win_step in _windows:
    pars = dict(
        window_length=win_len,
        window_step=win_step,
    )
    test_kwargs.append(pars)

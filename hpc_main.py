import subprocess
from os import remove

import numpy as np

import config
from BCISystem import BCISystem, Databases, XvalidateMethod, FeatureType
from preprocess import save_to_json, load_from_json, OfflineDataPreprocessor, init_base_config

CHECKPOINT = 'checkpoint.json'
FEATURE_DIR = 'feature_dir'
SUBJECT_NUM = 'subj_num'
cp_info = dict()


def make_test(subject_from, feature, db_name, verbose=False, subj_n_fold_num=5,
              epoch_tmin=0, epoch_tmax=4,
              window_length=1, window_step=.1,
              use_drop_subject_list=True,
              classifier_kwargs=None, filter_params=None):
    if filter_params is None:
        filter_params = {}
    if classifier_kwargs is None:
        classifier_kwargs = {}

    file_name = 'log_{}_{}.csv'.format(feature['feature_type'].name, subject_from)

    # generate database if not available
    proc = OfflineDataPreprocessor(
        init_base_config(),
        epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
        window_length=window_length, window_step=window_step,
        fast_load=fast_load, filter_params=filter_params,
        use_drop_subject_list=use_drop_subject_list,
    )
    proc.use_db(db_name)
    subject_list = np.arange(subject_from, proc.get_subject_num() + 1)
    proc.run(subject_list, **feature)

    # make test
    bci = BCISystem(make_logs=True, verbose=verbose)
    for subject in subject_list:
        cp_info[SUBJECT_NUM] = subject
        bci.offline_processing(db_name=db_name,
                               feature_params=feature,
                               fast_load=True,
                               epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
                               window_length=window_length, window_step=window_step,
                               filter_params=filter_params,
                               method=XvalidateMethod.SUBJECT,
                               subject_list=subject,
                               use_drop_subject_list=use_drop_subject_list,
                               subj_n_fold_num=subj_n_fold_num,
                               classifier_kwargs=classifier_kwargs)
        bci.log_results(file_name)
        save_to_json(CHECKPOINT, cp_info)


if __name__ == '__main__':
    user = subprocess.check_output('whoami').decode('utf-8').strip('\n')
    fast_load = True
    try:
        cp_info = load_from_json(CHECKPOINT)
    except FileNotFoundError:
        cp_info[FEATURE_DIR] = '/scratch{}/bci_{}'.format(np.random.randint(1, 5), user)
        cp_info[SUBJECT_NUM] = 1
        fast_load = False

    config.DIR_FEATURE_DB = cp_info[FEATURE_DIR]

    make_test(cp_info[SUBJECT_NUM], feature=FeatureType.FFT_POWER, db_name=Databases.PHYSIONET)

    remove(CHECKPOINT)

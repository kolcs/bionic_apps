import subprocess
from datetime import datetime
from os import remove
from shutil import rmtree

import numpy as np

from BCISystem import BCISystem, Databases, XvalidateMethod, FeatureType, ClassifierType
from preprocess import save_to_json, load_from_json, OfflineDataPreprocessor, init_base_config, ioprocess

CHECKPOINT = 'checkpoint.json'
FEATURE_DIR = 'feature_dir'
SUBJECT_NUM = 'subj_num'
cp_info = dict()


def make_test(feature_params, db_name, subject_from=1, verbose=True,
              method=XvalidateMethod.SUBJECT, subj_n_fold_num=5,
              epoch_tmin=0, epoch_tmax=4,
              window_length=1, window_step=.1,
              use_drop_subject_list=True, fast_load=False,
              classifier_type=ClassifierType.SVM, classifier_kwargs=None, filter_params=None,
              validation_split=0, log_file=None, **kwargs):
    if log_file is None:
        log_file = 'log_{}_{}.csv'.format(feature_params['feature_type'].name, datetime.now().strftime("%Y%m%d-%H%M%S"))

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
    proc_subjects = set(np.array(subject_list).flatten()) if subject_list is not None else None
    proc.run(proc_subjects, **feature_params)

    # make test
    bci = BCISystem(make_logs=True, verbose=verbose, log_file=log_file)
    for subject in subject_list:
        subject = int(subject)
        if len(cp_info) > 0:
            cp_info[SUBJECT_NUM] = subject
            save_to_json(CHECKPOINT, cp_info)

        bci.offline_processing(db_name=db_name,
                               feature_params=feature_params,
                               fast_load=True,
                               epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
                               window_length=window_length, window_step=window_step,
                               filter_params=filter_params,
                               method=method,
                               subject_list=subject,
                               use_drop_subject_list=use_drop_subject_list,
                               subj_n_fold_num=subj_n_fold_num,
                               classifier_type=classifier_type,
                               classifier_kwargs=classifier_kwargs,
                               validation_split=validation_split,
                               **kwargs)


def hpc_run_cp(checkpoint=None, verbose=False, test_func=make_test, **test_kwargs):
    global cp_info, CHECKPOINT
    if type(checkpoint) is str:
        CHECKPOINT = checkpoint

    user = subprocess.check_output('whoami').decode('utf-8').strip('\n')
    fast_load = True
    try:
        cp_info = load_from_json(CHECKPOINT)
    except FileNotFoundError:
        path = '/scratch{}/bci_{}'.format(np.random.randint(1, 5), user)
        rmtree(path, ignore_errors=True)
        cp_info[FEATURE_DIR] = path
        cp_info[SUBJECT_NUM] = 1
        fast_load = False
    ioprocess.DIR_FEATURE_DB = cp_info[FEATURE_DIR]

    # running the test with checkpoints...
    test_func(subject_from=cp_info[SUBJECT_NUM], fast_load=fast_load,
              verbose=verbose, **test_kwargs)

    remove(CHECKPOINT)


def hpc_run_nocp(test_func, **test_kwargs):
    user = subprocess.check_output('whoami').decode('utf-8').strip('\n')
    tmp_data_path = '/scratch{}/bci_{}'.format(np.random.randint(1, 5), user)
    ioprocess.DIR_FEATURE_DB = tmp_data_path
    test_func(**test_kwargs)
    # rmtree(tmp_data_path, ignore_errors=True)


if __name__ == '__main__':
    # This is an example how to run HPC code:
    hpc_kwargs = dict(
        db_name=Databases.GAME_PAR_D,
        method=XvalidateMethod.SUBJECT,
        filter_params=dict(
            # order=5, l_freq=1, h_freq=None
        ),
        feature_params=dict(
            feature_type=FeatureType.AVG_FFT_POWER,
            fft_low=7, fft_high=14
        ),
        classifier_type=ClassifierType.SVM,
        classifier_kwargs=dict(
            # weights=None,
            # batch_size=32,
            # epochs=10,
        ),
        validation_split=0
    )
    hpc_run_cp(**hpc_kwargs)

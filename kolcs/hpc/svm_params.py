import subprocess
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np

from BCISystem import BCISystem, Databases, XvalidateMethod, FeatureType, ClassifierType
from hpc_main import hpc_run_nocp
from preprocess import DataLoader

LOG_TYPE = ''
LOG_DIR = 'svm_params'

hpc_kwargs = dict(
    method=XvalidateMethod.SUBJECT,
    epoch_tmin=0, epoch_tmax=4,
    window_length=2, window_step=.1,
    filter_params=dict(
        order=5, l_freq=1, h_freq=45
    ),
    classifier_type=ClassifierType.SVM,

    subj_n_fold_num=5,
    use_drop_subject_list=True,
    validation_split=0,
    do_artefact_rejection=True,
    mimic_online_method=False,
    make_channel_selection=False,
)


class SVMConfig(Enum):
    RBF = 'rbf_tuning'
    POLYNOMIAL = 'poly'
    LINEAR = 'lin'


def make_subject_test(*args, **kwargs):
    lin_test(*args, **kwargs)
    poly_test(*args, **kwargs)
    rbf_test(*args, **kwargs)


def rbf_test(db_name, subj, verbose=True,
             method=XvalidateMethod.SUBJECT, subj_n_fold_num=5,
             epoch_tmin=0, epoch_tmax=4,
             window_length=1, window_step=.1,
             use_drop_subject_list=True,
             classifier_type=ClassifierType.SVM, filter_params=None,
             validation_split=0, folder=Path(), **kwargs):
    log_file = str(folder.joinpath('rbf_subj{}_{}.csv'.format(subj, datetime.now().strftime("%Y%m%d-%H%M%S"))))
    bci = BCISystem(make_logs=True, verbose=verbose, log_file=log_file)
    for fft_low in range(2, 39, 2):
        for fft_high in range(fft_low + 2, min(fft_low + 20, 41), 2):
            feature_extraction = dict(
                feature_type=FeatureType.AVG_FFT_POWER,
                fft_low=fft_low, fft_high=fft_high,
            )
            for c in 2 ** np.linspace(-7, 17, num=12):
                for gamma in 2 ** np.linspace(-17, 7, num=12):
                    classifier_kwargs = dict(
                        C=c, gamma=gamma, cache_size=400
                    )

                    bci.offline_processing(db_name=db_name,
                                           feature_params=feature_extraction,
                                           epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
                                           window_length=window_length, window_step=window_step,
                                           filter_params=filter_params,
                                           method=method,
                                           subject_list=subj,
                                           fast_load=True,
                                           use_drop_subject_list=use_drop_subject_list,
                                           subj_n_fold_num=subj_n_fold_num,
                                           classifier_type=classifier_type,
                                           classifier_kwargs=classifier_kwargs,
                                           validation_split=validation_split,
                                           **kwargs)


def poly_test(db_name, subj, verbose=True,
              method=XvalidateMethod.SUBJECT, subj_n_fold_num=5,
              epoch_tmin=0, epoch_tmax=4,
              window_length=1, window_step=.1,
              use_drop_subject_list=True,
              classifier_type=ClassifierType.SVM, filter_params=None,
              validation_split=0, folder=Path(), **kwargs):
    log_file = str(folder.joinpath('poly_subj{}_{}.csv'.format(subj, datetime.now().strftime("%Y%m%d-%H%M%S"))))
    bci = BCISystem(make_logs=True, verbose=verbose, log_file=log_file)
    for fft_low in range(2, 39, 2):
        for fft_high in range(fft_low + 2, min(fft_low + 20, 41), 2):
            feature_extraction = dict(
                feature_type=FeatureType.AVG_FFT_POWER,
                fft_low=fft_low, fft_high=fft_high,
            )
            for c in 2 ** np.linspace(-7, 17, num=12):
                for gamma in 2 ** np.linspace(-17, 7, num=12):
                    classifier_kwargs = dict(
                        C=c, kernel='poly', gamma=gamma, cache_size=400
                    )

                    bci.offline_processing(db_name=db_name,
                                           feature_params=feature_extraction,
                                           epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
                                           window_length=window_length, window_step=window_step,
                                           filter_params=filter_params,
                                           method=method,
                                           subject_list=subj,
                                           fast_load=True,
                                           use_drop_subject_list=use_drop_subject_list,
                                           subj_n_fold_num=subj_n_fold_num,
                                           classifier_type=classifier_type,
                                           classifier_kwargs=classifier_kwargs,
                                           validation_split=validation_split,
                                           **kwargs)


def lin_test(db_name, subj, verbose=True,
             method=XvalidateMethod.SUBJECT, subj_n_fold_num=5,
             epoch_tmin=0, epoch_tmax=4,
             window_length=1, window_step=.1,
             use_drop_subject_list=True,
             classifier_type=ClassifierType.SVM, filter_params=None,
             validation_split=0, folder=Path(), **kwargs):
    log_file = str(folder.joinpath('lin_subj{}_{}.csv'.format(subj, datetime.now().strftime("%Y%m%d-%H%M%S"))))
    bci = BCISystem(make_logs=True, verbose=verbose, log_file=log_file)
    for fft_low in range(2, 39, 2):
        for fft_high in range(fft_low + 2, min(fft_low + 20, 41), 2):
            feature_extraction = dict(
                feature_type=FeatureType.AVG_FFT_POWER,
                fft_low=fft_low, fft_high=fft_high,
            )
            for c in 2 ** np.linspace(-7, 17, num=12):
                classifier_kwargs = dict(
                    C=c, kernel='linear', cache_size=400
                )

                bci.offline_processing(db_name=db_name,
                                       feature_params=feature_extraction,
                                       epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
                                       window_length=window_length, window_step=window_step,
                                       filter_params=filter_params,
                                       method=method,
                                       subject_list=subj,
                                       fast_load=True,
                                       use_drop_subject_list=use_drop_subject_list,
                                       subj_n_fold_num=subj_n_fold_num,
                                       classifier_type=classifier_type,
                                       classifier_kwargs=classifier_kwargs,
                                       validation_split=validation_split,
                                       **kwargs)


def make_one_test():
    _, db_name, subject = sys.argv

    folder = Path(LOG_DIR).joinpath(db_name, LOG_TYPE)
    folder.mkdir(parents=True, exist_ok=True)

    additional_kwargs = dict(
        db_name=Databases(db_name),
        subj=int(subject),
        folder=folder
    )
    make_test = make_subject_test
    hpc_kwargs.update(additional_kwargs)
    hpc_run_nocp(make_test, **hpc_kwargs)


# stuff to script:
# python -c "from kolcs.hpc.svm_params import start_svm_test; start_svm_test()"


def start_svm_test():  # call this from script of from python
    job_list = 'Submitted batch jobs'
    Path(LOG_DIR).joinpath('out').mkdir(parents=True, exist_ok=True)  # sdt out and error files
    for db_name in [Databases.PHYSIONET, Databases.TTK]:
        loader = DataLoader().use_db(db_name)
        for subj in loader.get_subject_list():
            subj = int(subj)
            cmd = f'sbatch {LOG_DIR}/'
            cmd += 'single_cpu.sh'
            cmd += f' {__file__} {db_name.value} {subj}'
            ans = subprocess.check_output(cmd, shell=True)
            job_list += ' ' + ans.decode('utf-8').strip('\n').strip('\r').strip('Submitted batch job')
    print(job_list)


if __name__ == '__main__':
    make_one_test()

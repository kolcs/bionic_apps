import subprocess
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path

from BCISystem import Databases, XvalidateMethod, FeatureType, ClassifierType
from hpc_main import hpc_run_cp, hpc_run_nocp, make_test

LOG_TYPE = ''
LOG_DIR = 'big_test'

hpc_kwargs = dict(
    method=XvalidateMethod.SUBJECT,
    epoch_tmin=0, epoch_tmax=4,
    window_length=1, window_step=.1,
    filter_params=dict(
        # order=5, l_freq=1, h_freq=None
    ),
    classifier_type=ClassifierType.SVM,
    classifier_kwargs=dict(
        # weights=None,
        # batch_size=32,
        # epochs=10,
    ),
    subj_n_fold_num=5,
    use_drop_subject_list=True,
    validation_split=0,
)


class TestFeature(Enum):
    ALPHA = 'alpha'
    BETA = 'beta'
    RANGE30 = 'range30'
    RANGE40 = 'range40'
    MULTI = 'multi'


def get_feature_params(test_feature):
    assert test_feature in TestFeature, 'Only TestFeature is accepted.'
    if test_feature is TestFeature.ALPHA:
        feature_extraction = dict(
            feature_type=FeatureType.AVG_FFT_POWER,
            fft_low=7, fft_high=14
        )
    elif test_feature is TestFeature.BETA:
        feature_extraction = dict(
            feature_type=FeatureType.AVG_FFT_POWER,
            fft_low=14, fft_high=30
        )
    elif test_feature is TestFeature.RANGE30:
        feature_extraction = dict(
            feature_type=FeatureType.FFT_RANGE,
            fft_low=2, fft_high=30, fft_step=2, fft_width=2
        )
    elif test_feature is TestFeature.RANGE40:
        feature_extraction = dict(
            feature_type=FeatureType.FFT_RANGE,
            fft_low=2, fft_high=40, fft_step=2, fft_width=2
        )
    elif test_feature is TestFeature.MULTI:
        fft_ranges = [(14, 36), (18, 32), (18, 36), (22, 28),
                      (22, 36), (26, 32), (26, 36)]

        feature_extraction = dict(
            feature_type=FeatureType.MULTI_AVG_FFT_POW,
            fft_ranges=fft_ranges
        )
    else:
        raise NotImplementedError
    return feature_extraction


def make_one_test():
    _, db_name, test_feature = sys.argv

    folder = Path(LOG_DIR).joinpath(db_name, LOG_TYPE)
    folder.mkdir(parents=True, exist_ok=True)
    file = str(folder.joinpath('{}-{}.csv'.format(test_feature, datetime.now().strftime("%Y%m%d-%H%M%S"))))

    additional_kwargs = dict(
        db_name=Databases(db_name),
        feature_params=get_feature_params(TestFeature(test_feature)),
        log_file=file,
    )
    hpc_kwargs.update(additional_kwargs)
    hpc_run_nocp(make_test, **hpc_kwargs)


# stuff to script:
# python -c "from kolcs.hpc.big_test import start_big_test; start_big_test()"


def start_big_test():  # call this from script of from python
    job_list = 'Submitted batch jobs'
    Path(LOG_DIR).joinpath('out').mkdir(parents=True, exist_ok=True)  # sdt out and error files
    for db_name in [Databases.PHYSIONET, Databases.TTK]:
        for feature in [TestFeature.ALPHA, TestFeature.BETA,
                        TestFeature.RANGE30,
                        TestFeature.RANGE40,
                        TestFeature.MULTI
                        ]:
            cmd = f'sbatch {LOG_DIR}/'
            if feature in [TestFeature.RANGE40, TestFeature.RANGE30, TestFeature.MULTI]:
                cmd += 'multi_cpu.sh'
            else:
                cmd += 'single_cpu.sh'
            cmd += f' {__file__} {db_name.value} {feature.value}'
            ans = subprocess.check_output(cmd, shell=True)
            job_list += ' ' + ans.decode('utf-8').strip('\n').strip('\r').strip('Submitted batch job')
    print(job_list)


if __name__ == '__main__':
    make_one_test()

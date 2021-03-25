from time import time, sleep
from warnings import warn

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from BCISystem import BCISystem, FeatureType, XvalidateMethod

SUM_NAME = 'weight_sum'


def _generate_table(eeg, filter_list, acc_from=.6, acc_to=1, acc_diff=.01):
    d = pd.DataFrame(eeg[filter_list].groupby(filter_list).count())  # data permutation

    new_cols = list()
    for flow, fhigh in [(acc, acc + acc_diff) for acc in np.arange(acc_from, acc_to, acc_diff)[::-1]]:
        new_cols.append(np.round(flow, 3))
        d[np.round(flow, 3)] = \
            eeg[(eeg['Avg. Acc'] >= flow) & (eeg['Avg. Acc'] < fhigh)].groupby(filter_list, sort=True).count()[
                'Avg. Acc']

    d = d.fillna(0)
    d[SUM_NAME] = np.sum(
        np.array([d[col].array * (col - min(new_cols)) / (max(new_cols) - min(new_cols)) for col in new_cols]),
        axis=0)

    new_cols.insert(0, SUM_NAME)
    d = d.sort_values(new_cols, ascending=[False] * len(new_cols))
    return d


def _make_one_fft_test(eeg_file, db_name, fft_low, fft_high, classifier_kwargs=None):
    if classifier_kwargs is None:
        classifier_kwargs = {}

    bci = BCISystem(make_logs=True)
    feature_kwargs = dict(
        feature_type=FeatureType.AVG_FFT_POWER,
        fft_low=fft_low, fft_high=fft_high,
    )
    bci.offline_processing(
        db_name=db_name,
        feature_params=feature_kwargs,
        subject_list=0,
        method=XvalidateMethod.SUBJECT,
        epoch_tmin=0, epoch_tmax=4,
        window_length=1, window_step=.1,
        subj_n_fold_num=5,
        train_file=eeg_file,
        classifier_kwargs=classifier_kwargs
    )
    return bci._df


def parallel_search_for_fft_params(eeg_file, db_name, fft_search_min, fft_search_max, fft_search_step, best_n_fft,
                                   classifier_kwargs=None):
    """

    Parameters
    ----------
    eeg_file : str
        EEG file name.
    db_name : Databases
        Database type.
    fft_search_min, fft_search_max, fft_search_step : float
        Parameters for best FFT Power range search.
    best_n_fft : int
        The number of FFT Power ranges which will be selected.
    best_n_fft :
        Selecting the best n fft range from parameter selection. It is suggested to
        make it odd.
    classifier_kwargs : dict
         Arbitrary keyword arguments for classifier.

    Returns
    -------
    list of (float, float)
        A list of frequency ranges. Each each element of the list is a tuple, where the
        first element of a tuple corresponds to fft_min and the second is fft_max.

    """
    if classifier_kwargs is None:
        classifier_kwargs = {}

    params = list()
    for fft_low in np.arange(fft_search_min, fft_search_max - 2, fft_search_step):
        for fft_high in np.arange(fft_low + 2, fft_search_max, fft_search_step):
            params.append((eeg_file, db_name, fft_low, fft_high, classifier_kwargs))

    tic = time()
    res = Parallel(n_jobs=-2)(delayed(_make_one_fft_test)(*par) for par in params)
    print('Parameter selection took {:.2f} min'.format((time() - tic) / 60))

    # calculate results
    res = pd.concat(res)
    res = _generate_table(res[['FFT low', 'FFT high', 'Avg. Acc']],
                          ['FFT low', 'FFT high'])
    assert not res.empty, 'Result of _generate_table() can not be empty!'

    n_fft = min(len(res.index), best_n_fft)
    if n_fft < best_n_fft:
        warn('Only {} fft params were selected instead of {}'.format(n_fft, best_n_fft))
        sleep(1.5)
    fft_list = [res.index[i] for i in range(n_fft)]

    return fft_list

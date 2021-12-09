import mne
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils import class_weight

from preprocess import DataLoader, get_epochs_from_raw, SubjectHandle, standardize_channel_names, \
    Databases
from preprocess.artefact_faster import ArtefactFilter
from preprocess.ioprocess import _create_binary_label
from sklearn_pipeline_bci.feature_extraction import to_micro_volt, FFTCalc, AvgFFTCalc, get_fft_ranges, FeatureType


def _windowed_view(data, window_length, window_step):
    """Windower method which windows a given signal in to a given window size.
    Parameters
    ----------
    data : ndarray
        Data to be windowed. Shape: (n_channels, time)
    window_length : int
        Required window length in sample points.
    window_step : int
        Required window step in sample points.
    Returns
    -------
    ndarray
        Windowed data with shape (n_windows, n_channels, time)
    """
    overlap = window_length - window_step
    new_shape = ((data.shape[-1] - overlap) // window_step, data.shape[0], window_length)
    new_strides = (window_step * data.strides[-1], *data.strides)
    result = np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=new_strides)
    return result


def window_epochs(data, window_length, window_step, fs):
    """Create sliding windowed data from epochs.
    Parameters
    ----------
    data : ndarray
        Epochs data with shape (n_epochs, n_channels, time)
    window_length : float
        Length of sliding window in seconds.
    window_step : float
        Step of sliding window in seconds.
    fs : int
        Sampling frequency.
    Returns
    -------
    ndarray
        Windowed epochs data with shape (n_epochs, n_windows, n_channels, time)
    """
    windowed_tasks = []
    for i in range(len(data)):
        x = _windowed_view(data[i, :, :], int(window_length * fs), int(window_step * fs))
        windowed_tasks.append(np.array(x))
    return np.array(windowed_tasks)


def filter_raw(raw, f_type='butter', order=5, l_freq=1, h_freq=None):
    iir_params = dict(order=order, ftype=f_type, output='sos')
    raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir', iir_params=iir_params, skip_by_annotation='edge')
    return raw


def new_multi_svm(fs, fft_ranges):
    inner_clfs = [(f'unit{i}', make_pipeline(AvgFFTCalc(fft_low, fft_high),
                                             StandardScaler(), SVC()))
                  for i, (fft_low, fft_high) in enumerate(fft_ranges)]

    clf = make_pipeline(
        FunctionTransformer(to_micro_volt),
        FFTCalc(fs),
        VotingClassifier(inner_clfs, n_jobs=len(inner_clfs)) if len(inner_clfs) > 1 else inner_clfs[0][1]
    )

    return clf


def all_alpha_svm(fs, fft_low, fft_high):
    # todo: make all comb
    # - psd, psd2, fftabs, fftpow
    # - no norm, l2 norm, ...
    parallel_lines = [(f'unit{i}', make_pipeline(, StandardScaler()))]

    clf = make_pipeline(
        FunctionTransformer(to_micro_volt),
        FFTCalc(fs),
        AvgFFTCalc(fft_low, fft_high),
        FeatureUnion(parallel_lines),
        # todo: pca
        SVC()
    )
    return clf


def train_test_model(x, y, groups, fs, fft_ranges):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    kfold = StratifiedGroupKFold(shuffle=False)

    cross_acc = []
    for train, test in kfold.split(np.arange(len(x)), y, groups):
        train_x = x[train]
        train_y = y[train]
        test_x = x[test]
        test_y = y[test]

        clf = new_multi_svm(fs, fft_ranges)

        clf.fit(train_x, train_y)
        y_pred = clf.predict(test_x)
        y_pred = label_encoder.inverse_transform(y_pred)
        test_y = label_encoder.inverse_transform(test_y)

        # https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-and-f-measures
        class_report = classification_report(test_y, y_pred)
        conf_matrix = confusion_matrix(test_y, y_pred)
        acc = accuracy_score(test_y, y_pred)
        print(class_report)
        print("Confusion matrix:\n%s\n" % conf_matrix)
        print("Accuracy score: {}\n".format(acc))

        cross_acc.append(acc)
    print("Accuracy scores for k-fold crossvalidation: {}\n".format(cross_acc))
    print(f"Avg accuracy: {np.mean(cross_acc):.4f}   +/- {np.std(cross_acc):.4f}")
    return cross_acc


def test_db(feature_params, db_name,
            epoch_tmin=0, epoch_tmax=4,
            window_length=2, window_step=.1,
            use_drop_subject_list=True,
            filter_params=None,
            do_artefact_rejection=True,
            log_file='out.csv'):
    if filter_params is None:
        filter_params = {}
    loader = DataLoader('..', use_drop_subject_list=use_drop_subject_list,
                        subject_handle=SubjectHandle.INDEPENDENT_DAYS)
    loader.use_db(db_name)
    res = {
        'Subject': [],
        'Accuracy list': [],
        'Std of Avg. Acc': [],
        'Avg. Acc': []
    }

    for subj in loader.get_subject_list():
        files = loader.get_filenames_for_subject(subj)
        task_dict = loader.get_task_dict()
        event_id = loader.get_event_id()
        print(f'\nSubject{subj}')
        raw = mne.io.concatenate_raws([mne.io.read_raw(file) for file in files])
        raw.load_data()
        fs = raw.info['sfreq']

        standardize_channel_names(raw)
        try:  # check available channel positions
            mne.channels.make_eeg_layout(raw.info)
        except RuntimeError:  # if no channel positions are available create them from standard positions
            montage = mne.channels.make_standard_montage('standard_1005')  # 'standard_1020'
            raw.set_montage(montage, on_missing='warn')

        if len(filter_params) > 0:
            raw = filter_raw(raw, **filter_params)

        epochs = get_epochs_from_raw(raw, task_dict,
                                     epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
                                     event_id=event_id)

        if do_artefact_rejection:
            epochs = ArtefactFilter(apply_frequency_filter=False).offline_filter(epochs)

        ep_labels = [list(epochs[i].event_id)[0] for i in range(len(epochs))]

        if db_name is Databases.GAME_PAR_D:
            ep_labels = [_create_binary_label(label) for label in ep_labels]

        # window the epochs
        windowed_data = window_epochs(epochs.get_data(),
                                      window_length=window_length, window_step=window_step,
                                      fs=fs)
        groups = [i // windowed_data.shape[1] for i in range(windowed_data.shape[0] * windowed_data.shape[1])]
        labels = [ep_labels[i // windowed_data.shape[1]] for i in range(len(groups))]
        windowed_data = np.vstack(windowed_data)

        # features = FeatureExtractor(fs=fs, **feature_params).run(windowed_data)
        features = windowed_data
        fft_ranges = get_fft_ranges(**feature_params)

        print("####### Classification report for subject{}: #######".format(subj))
        cross_acc = train_test_model(features, labels, groups, fs, fft_ranges)
        res['Subject'].append(subj)
        res['Accuracy list'].append(cross_acc)
        res['Std of Avg. Acc'].append(np.std(cross_acc))
        res['Avg. Acc'].append(np.mean(cross_acc))
        pd.DataFrame(res).to_csv(log_file, sep=';', encoding='utf-8', index=False)


def run_new_multi_svm():
    test_db(
        feature_params=dict(
            feature_type=FeatureType.FFT_RANGE,
            fft_low=4, fft_high=30
        ),
        db_name=Databases.PHYSIONET,
        filter_params=dict(  # required for FASTER artefact filter
            order=5, l_freq=1, h_freq=45
        ),
        do_artefact_rejection=True,
        log_file='out.csv'
    )


if __name__ == '__main__':
    run_new_multi_svm()

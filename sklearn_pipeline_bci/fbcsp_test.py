import mne
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC

from preprocess import DataLoader, get_epochs_from_raw, SubjectHandle, standardize_channel_names, \
    Databases
from preprocess.artefact_faster import ArtefactFilter
from preprocess.ioprocess import _create_binary_label
from sklearn_pipeline_bci.fbcsp import FBCSP, FilterBank, WindowEpochs
from sklearn_pipeline_bci.utils import filter_mne_obj, balance_epoch_nums


def train_test_model(x, y, **window_kwargs):
    kfold = StratifiedKFold(n_splits=5, shuffle=True)

    n_epochs = len(x)
    x = FilterBank().transform(x)

    cross_acc = []
    for train, test in kfold.split(np.arange(n_epochs), y):
        train_x = x[:, train, ...]
        train_y = y[train]
        test_x = x[:, test, ...]
        test_y = y[test]

        fbcsp = FBCSP().fit(train_x, train_y)  # fit FBCSP on whole epochs

        # fbcsp = mne.decoding.CSP(n_components=4, log=True, cov_est='epoch')
        # train_x = train_x.get_data()
        # test_x = test_x.get_data()
        # fbcsp.fit(train_x, train_y)  # fit FBCSP on whole epochs

        windower = WindowEpochs(**window_kwargs, shuffle=True)

        train_x, train_y, _ = windower.transform(train_x, train_y)
        train_x = fbcsp.transform(train_x)
        test_x, test_y, _ = windower.transform(test_x, test_y)
        test_x = fbcsp.transform(test_x)

        clf = make_pipeline(
            # SelectKBest(mutual_info_classif, k=10),
            StandardScaler(),
            SVC(cache_size=1000)
        )

        clf.fit(train_x, train_y)
        y_pred = clf.predict(test_x)

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


def _select_fb(x, i):
    return x[:, i, ...]


def train_test_voting_FBCSP(x, y, **window_kwargs):
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    n_epochs = len(x)
    x = FilterBank().transform(x)
    n_filter = len(x)
    x = x.transpose((1, 0, 2, 3))

    cross_acc = []
    for train, test in kfold.split(np.arange(n_epochs), y):
        train_x = x[train]
        train_y = y[train]
        test_x = x[test]
        test_y = y[test]

        clfs = [(f'CSP {i}', make_pipeline(
            FunctionTransformer(_select_fb, kw_args=dict(i=i)),
            mne.decoding.UnsupervisedSpatialFilter(PCA(n_components=32)),
            mne.decoding.CSP(n_components=7, reg='ledoit_wolf', log=True),
            StandardScaler(),
            SVC(probability=True)
        )) for i in range(n_filter)]

        windower = WindowEpochs(**window_kwargs, shuffle=True)

        # train_x, train_y, _ = windower.transform(train_x, train_y)
        # train_x = fbcsp.transform(train_x)
        test_x = test_x.transpose((1, 0, 2, 3))
        test_x, test_y, _ = windower.transform(test_x, test_y)
        test_x = test_x.transpose((1, 0, 2, 3))
        # test_x = fbcsp.transform(test_x)

        # clf = VotingClassifier(clfs, voting='soft', n_jobs=len(clfs))
        clf = StackingClassifier(clfs, SVC(), n_jobs=len(clfs))

        clf.fit(train_x, train_y)
        y_pred = clf.predict(test_x)

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


def test_fbcsp_db(db_name,
                  epoch_tmin=0, epoch_tmax=4,
                  window_length=2, window_step=.1,
                  use_drop_subject_list=True,
                  filter_params=None,
                  do_artefact_rejection=True,
                  balance_data=True,
                  subj_cp=0,
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
        if subj < subj_cp:
            continue
        files = loader.get_filenames_for_subject(subj)
        task_dict = loader.get_task_dict()
        event_id = loader.get_event_id()
        print(f'\nSubject{subj}')
        raws = [mne.io.read_raw(file) for file in files]
        raw = mne.io.concatenate_raws(raws)
        raw.load_data()
        fs = raw.info['sfreq']

        standardize_channel_names(raw)
        try:  # check available channel positions
            mne.channels.make_eeg_layout(raw.info)
        except RuntimeError:  # if no channel positions are available create them from standard positions
            montage = mne.channels.make_standard_montage('standard_1005')  # 'standard_1020'
            raw.set_montage(montage, on_missing='warn')

        if len(filter_params) > 0:
            raw = filter_mne_obj(raw, **filter_params)

        epochs = get_epochs_from_raw(raw, task_dict,
                                     epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
                                     event_id=event_id)
        del raw

        if do_artefact_rejection:
            epochs = ArtefactFilter(apply_frequency_filter=False).offline_filter(epochs)

        ep_labels = [list(epochs[i].event_id)[0] for i in range(len(epochs))]

        if balance_data:
            epochs, ep_labels = balance_epoch_nums(epochs, ep_labels)

        if db_name is Databases.GAME_PAR_D:
            ep_labels = [_create_binary_label(label) for label in ep_labels]

        print("####### Classification report for subject{}: #######".format(subj))
        # cross_acc = train_test_model(epochs, ep_labels,
        #                              window_length=window_length,
        #                              window_step=window_step,
        #                              fs=fs)
        cross_acc = train_test_voting_FBCSP(epochs, ep_labels,
                                            window_length=window_length,
                                            window_step=window_step,
                                            fs=fs)
        res['Subject'].append(subj)
        res['Accuracy list'].append(cross_acc)
        res['Std of Avg. Acc'].append(np.std(cross_acc))
        res['Avg. Acc'].append(np.mean(cross_acc))
        pd.DataFrame(res).to_csv(log_file, sep=';', encoding='utf-8', index=False)


if __name__ == '__main__':
    filter_params = dict(  # required for FASTER artefact filter
        order=5,
        l_freq=1,
        h_freq=45
    )
    test_fbcsp_db(Databases.PHYSIONET, filter_params=filter_params,
                  log_file='Stacked_classifiers.csv')

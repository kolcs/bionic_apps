import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from .ai.classifier import test_classifier, init_classifier, ClassifierType
from .ai.interface import TFBaseNet
from .config import SAVE_PATH
from .databases import EEG_Databases
from .feature_extraction import FeatureType
from .handlers import init_hdf5_db, ResultHandler
from .model_selection import BalancedKFold
from .preprocess import generate_eeg_db
from .preprocess.io import SubjectHandle
from .utils import save_pickle_data
from .validations import validate_feature_classifier_pair

DB_FILE = SAVE_PATH.joinpath('database.hdf5')


def train_test_data(classifier_type, x, y, groups, lab_enc,
                    *, n_splits=5, shuffle=False,
                    epochs=None, save_classifiers=False, **classifier_kwargs):
    kfold = BalancedKFold(n_splits=n_splits, shuffle=shuffle)
    cross_acc = list()
    saved_clf_names = list()
    for i, (train, test) in enumerate(kfold.split(y=y, groups=groups)):
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]

        clf = init_classifier(classifier_type, x[0].shape, len(lab_enc.classes_), **classifier_kwargs)

        if epochs is None:
            clf.fit(x_train, y_train)
        else:
            clf.fit(x_train, y_train, epochs=epochs)

        acc = test_classifier(clf, x_test, y_test, lab_enc)
        cross_acc.append(acc)

        if save_classifiers:
            if isinstance(clf, TFBaseNet):
                raise NotImplementedError()
            elif isinstance(clf, (BaseEstimator, Pipeline, ClassifierMixin)):
                save_path = SAVE_PATH.joinpath('sklearn', f'clf{i}.pkl')
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_pickle_data(save_path, clf)
                saved_clf_names.append(save_path)
            else:
                raise NotImplementedError()

    print(f"Accuracy scores for k-fold crossvalidation: {cross_acc}\n")
    print(f"Avg accuracy: {np.mean(cross_acc):.4f}   +/- {np.std(cross_acc):.4f}")
    if save_classifiers:
        return cross_acc, saved_clf_names
    return cross_acc


def make_within_subject_classification(db_filename, classifier_type, classifier_kwargs=None,
                                       n_splits=5, res_handler=None, save_res=True):
    if classifier_kwargs is None:
        classifier_kwargs = {}

    db, y_all, subj_ind, ep_ind, le, _ = init_hdf5_db(db_filename)

    for subj in np.unique(subj_ind):
        print(f'Subject{subj}')

        x = db.get_data(subj_ind == subj)
        y = y_all[subj_ind == subj]
        groups = ep_ind[subj_ind == subj]

        cross_acc = train_test_data(classifier_type, x, y, groups=groups, lab_enc=le,
                                    n_splits=n_splits, shuffle=True, **classifier_kwargs)

        if res_handler is not None:
            res_handler.add({'Subject': [f'Subject{subj}'],
                             'Accuracy list': [cross_acc],
                             'Std of Avg. Acc': [np.std(cross_acc)],
                             'Avg. Acc': [np.mean(cross_acc)]})
            if save_res:
                res_handler.save()

    if res_handler is not None:
        res_handler.print_db_res()
    db.close()


def test_eegdb_within_subject(
        db_name=EEG_Databases.PHYSIONET,
        feature_type=FeatureType.RAW,
        epoch_tmin=0, epoch_tmax=4,
        window_len=2, window_step=.1, *,
        feature_kwargs=None,
        use_drop_subject_list=True,
        filter_params=None,
        do_artefact_rejection=True,
        balance_data=True,
        subject_handle=SubjectHandle.INDEPENDENT_DAYS,
        n_splits=5,
        classifier_type=ClassifierType.ENSEMBLE,
        classifier_kwargs=None,
        # ch_mode='all', ep_mode='distinct',
        db_file=DB_FILE, log_file='out.csv', base_dir='.',
        save_res=True,
        fast_load=True,
):
    if classifier_kwargs is None:
        classifier_kwargs = {}

    feature_type, classifier_type = validate_feature_classifier_pair(feature_type, classifier_type)

    fix_params = dict(window_len=window_len, window_step=window_step,
                      n_splits=n_splits,
                      # ch_mode=ch_mode, ep_mode=ep_mode,
                      classifier=classifier_type.name)
    fix_params.update(classifier_kwargs)

    results = ResultHandler(fix_params, ['Subject', 'Accuracy list', 'Std of Avg. Acc', 'Avg. Acc'],
                            to_beginning=('Subject',), filename=log_file)

    generate_eeg_db(db_name, db_file, feature_type,
                    epoch_tmin, epoch_tmax,
                    window_len, window_step,
                    feature_kwargs=feature_kwargs,
                    use_drop_subject_list=use_drop_subject_list,
                    filter_params=filter_params,
                    do_artefact_rejection=do_artefact_rejection,
                    balance_data=balance_data,
                    subject_handle=subject_handle,
                    base_dir=base_dir, fast_load=fast_load)

    make_within_subject_classification(db_file, classifier_type,
                                       classifier_kwargs=classifier_kwargs,
                                       n_splits=n_splits, res_handler=results,
                                       save_res=save_res)

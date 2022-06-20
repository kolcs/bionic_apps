import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from tensorflow import data as tf_data

from .ai.classifier import test_classifier, init_classifier, ClassifierType
from .ai.interface import TFBaseNet
from .config import SAVE_PATH
from .databases import EEG_Databases
from .feature_extraction import FeatureType
from .handlers import ResultHandler, HDF5Dataset
from .handlers.tf import get_tf_dataset
from .model_selection import BalancedKFold, LeavePSubjectGroupsOutSequentially
from .preprocess import generate_eeg_db
from .preprocess.io import SubjectHandle
from .utils import save_pickle_data, mask_to_ind
from .validations import validate_feature_classifier_pair

DB_FILE = SAVE_PATH.joinpath('database.hdf5')


def train_test_subject_data(db, subj_ind, classifier_type,
                            *, n_splits=5, shuffle=False,
                            epochs=None, save_classifiers=False,
                            label_encoder=None, batch_size=32,
                            **classifier_kwargs):
    kfold = BalancedKFold(n_splits=n_splits, shuffle=shuffle)

    y = db.get_meta('y')
    ep_ind = db.get_meta('ep_group')[subj_ind]
    if label_encoder is None:
        label_encoder = LabelEncoder().fit(y)
    y = label_encoder.transform(y[subj_ind])

    cross_acc = list()
    saved_clf_names = list()
    for i, (train, test) in enumerate(kfold.split(y=y, groups=ep_ind)):
        y_test = y[test]

        clf = init_classifier(classifier_type, db.get_data(subj_ind[0]).shape, len(label_encoder.classes_),
                              **classifier_kwargs)

        if epochs is None:
            x = db.get_data(subj_ind)
            x_train = x[train]
            x_test = x[test]
            y_train = y[train]
            clf.fit(x_train, y_train)
        else:
            y = label_encoder.transform(db.get_meta('y'))
            tf_dataset = get_tf_dataset(db, y, subj_ind[train]).batch(batch_size)
            tf_dataset = tf_dataset.prefetch(tf_data.experimental.AUTOTUNE)
            clf.fit(tf_dataset, epochs=epochs)
            x_test = db.get_data(subj_ind[test])
            clf.evaluate(x_test, y_test)

        acc = test_classifier(clf, x_test, y_test, label_encoder)
        cross_acc.append(acc)

        if save_classifiers:
            if isinstance(clf, TFBaseNet):
                save_path = SAVE_PATH.joinpath('tensorflow', f'clf{i}.h5')
                save_path.parent.mkdir(parents=True, exist_ok=True)
                clf.save(save_path)
            elif isinstance(clf, (BaseEstimator, Pipeline, ClassifierMixin)):
                save_path = SAVE_PATH.joinpath('sklearn', f'clf{i}.pkl')
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_pickle_data(save_path, clf)
            else:
                raise NotImplementedError()
            saved_clf_names.append(save_path)

    print(f"Accuracy scores for k-fold crossvalidation: {cross_acc}\n")
    print(f"Avg accuracy: {np.mean(cross_acc):.4f}   +/- {np.std(cross_acc):.4f}")
    if save_classifiers:
        return cross_acc, saved_clf_names
    return cross_acc


def make_within_subject_classification(db_filename, classifier_type, classifier_kwargs=None,
                                       n_splits=5, res_handler=None, save_res=True):
    if classifier_kwargs is None:
        classifier_kwargs = {}

    db = HDF5Dataset(db_filename)
    all_subj = db.get_meta('subject')

    for subj in np.unique(all_subj):
        print(f'Subject{subj}')
        subj_ind = mask_to_ind(subj == all_subj)
        cross_acc = train_test_subject_data(db, subj_ind, classifier_type, n_splits=n_splits,
                                            shuffle=True, **classifier_kwargs)

        if res_handler is not None:
            res_handler.add({'Subject': [f'Subject{subj}'],
                             'Accuracy list': [cross_acc],
                             'Std of Avg. Acc': [np.std(cross_acc)],
                             'Avg. Acc': [np.mean(cross_acc)]})
            if save_res:
                res_handler.save()

    db.close()
    if res_handler is not None:
        res_handler.print_db_res()


def test_eegdb_within_subject(
        db_name=EEG_Databases.PHYSIONET,
        feature_type=FeatureType.HUGINES,
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


def make_cross_subject_classification(db_filename, classifier_type,
                                      leave_out_n_subjects=10, res_handler=None,
                                      save_res=True, epochs=None, batch_size=32,
                                      **classifier_kwargs):
    db = HDF5Dataset(db_filename)
    all_subj = db.get_meta('subject')
    y = db.get_meta('y')
    label_encoder = LabelEncoder().fit(y)
    y = label_encoder.transform(y)

    # todo: validation - subj / epoch level?, leave_out_n == 1 ?
    for train_ind, test_ind in LeavePSubjectGroupsOutSequentially(leave_out_n_subjects).split(groups=all_subj):
        clf = init_classifier(classifier_type, db.get_data(train_ind[0]).shape,
                              len(label_encoder.classes_), **classifier_kwargs)

        if epochs is None:
            x_train = db.get_data(train_ind)
            y_train = y[train_ind]
            clf.fit(x_train, y_train)
        else:
            tf_dataset = get_tf_dataset(db, y, train_ind).batch(batch_size)
            tf_dataset = tf_dataset.prefetch(tf_data.experimental.AUTOTUNE)
            clf.fit(tf_dataset, epochs=epochs)

        # test subjects individually - check network generalization capability
        for subj in np.unique(all_subj[test_ind]):
            print(f'Subject{subj}')
            test_subj_ind = mask_to_ind(subj == all_subj)

            x_test = db.get_data(test_subj_ind)
            y_test = y[test_subj_ind]
            acc = test_classifier(clf, x_test, y_test, label_encoder)

            if res_handler is not None:
                res_handler.add({'Subject': [f'Subject{subj}'],
                                 'Left out subjects': [np.unique(all_subj[test_ind])],
                                 'Accuracy': [acc]})
                if save_res:
                    res_handler.save()

    db.close()
    if res_handler is not None:
        res_handler.print_db_res(col='Accuracy')


def test_eegdb_cross_subject(
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
        leave_out_n_subjects=10,
        classifier_type=ClassifierType.EEG_NET,
        classifier_kwargs=None,
        db_file=DB_FILE, log_file='out.csv', base_dir='.',
        save_res=True,
        fast_load=True,
):
    if classifier_kwargs is None:
        classifier_kwargs = {}

    feature_type, classifier_type = validate_feature_classifier_pair(feature_type, classifier_type)

    fix_params = dict(window_len=window_len, window_step=window_step,
                      classifier=classifier_type.name)
    fix_params.update(classifier_kwargs)

    results = ResultHandler(fix_params,
                            ['Subject', 'Left out subjects', 'Accuracy'],
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

    make_cross_subject_classification(db_file, classifier_type,
                                      leave_out_n_subjects=leave_out_n_subjects,
                                      res_handler=results,
                                      save_res=save_res,
                                      **classifier_kwargs)

from pathlib import Path

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
from .utils import mask_to_ind, process_run, save_pickle_data
from .validations import validate_feature_classifier_pair

DB_FILE = SAVE_PATH.joinpath('database.hdf5')


def _get_train_val_ind(validation_split, train_groups):
    groups = np.unique(train_groups)
    val_num = int(groups.size * validation_split)
    if validation_split > 0 and val_num == 0:
        val_num = 1
    np.random.shuffle(groups)
    val_mask = np.in1d(train_groups, groups[:val_num])
    val_ind = mask_to_ind(val_mask)
    tr_ind = mask_to_ind(~val_mask)
    return tr_ind, val_ind


def _get_balanced_train_val_ind(validation_split, train_labels, train_groups):
    assert validation_split > 0, 'Validation split must be greater than 0!'
    min_split = min([len(np.unique(train_groups[label == train_labels])) for label in np.unique(train_labels)])
    n_splits = min(int(1 / validation_split), min_split)
    tr_ind, val_ind = next(BalancedKFold(n_splits, shuffle=True).split(y=train_labels, groups=train_groups))
    return tr_ind, val_ind


def _one_fold(train_ind, test_ind, subj_ind, ep_ind, y, db,
              *, shuffle, label_encoder, classifier_type,
              epochs=None, batch_size=32, validation_split=0.,
              save_classifier=False, i, save_path, **classifier_kwargs):
    if shuffle:
        np.random.shuffle(train_ind)
    y_train = y[train_ind]
    y_test = y[test_ind]

    clf = init_classifier(classifier_type, db.get_data(subj_ind[0]).shape, len(label_encoder.classes_),
                          **classifier_kwargs)

    if epochs is None:
        x = db.get_data(subj_ind)
        x_train = x[train_ind]
        x_test = x[test_ind]
        clf.fit(x_train, y_train)
    else:
        y = label_encoder.transform(db.get_y())
        if validation_split > 0:
            tr, val = _get_balanced_train_val_ind(validation_split, y_train, ep_ind[train_ind])
            train_tf_ds = get_tf_dataset(db, y, subj_ind[train_ind[tr]]).batch(batch_size)
            train_tf_ds = train_tf_ds.prefetch(tf_data.experimental.AUTOTUNE)
            val_tf_ds = get_tf_dataset(db, y, subj_ind[train_ind[val]]).batch(batch_size)
            val_tf_ds = val_tf_ds.cache()
            clf.fit(train_tf_ds, epochs=epochs, validation_data=val_tf_ds)
        else:
            tf_dataset = get_tf_dataset(db, y, subj_ind[train_ind]).batch(batch_size)
            tf_dataset = tf_dataset.prefetch(tf_data.experimental.AUTOTUNE)
            clf.fit(tf_dataset, epochs=epochs)
        x_test = db.get_data(subj_ind[test_ind])
        clf.evaluate(x_test, y_test)

    acc = test_classifier(clf, x_test, y_test, label_encoder)
    ans = acc

    if save_classifier:
        if isinstance(clf, TFBaseNet):
            file = save_path.joinpath('tensorflow', f'clf{i}.h5')
            file.parent.mkdir(parents=True, exist_ok=True)
            clf.save(file)
        elif isinstance(clf, (BaseEstimator, Pipeline, ClassifierMixin)):
            file = SAVE_PATH.joinpath('sklearn', f'clf{i}.pkl')
            file.parent.mkdir(parents=True, exist_ok=True)
            save_pickle_data(file, clf)
        else:
            raise NotImplementedError()
        ans = (acc, file)

    db.close()
    return ans


def train_test_subject_data(db, subj_ind, classifier_type,
                            *, n_splits=5, shuffle=False,
                            epochs=None, save_classifiers=False,
                            label_encoder=None, batch_size=32,
                            validation_split=.0,
                            **classifier_kwargs):
    kfold = BalancedKFold(n_splits=n_splits, shuffle=shuffle)

    y = db.get_y()
    ep_ind = db.get_epoch_group()[subj_ind]
    if label_encoder is None:
        label_encoder = LabelEncoder().fit(y)
    y = label_encoder.transform(y[subj_ind])

    cross_acc = list()
    saved_clf_names = list()
    for i, (train, test) in enumerate(kfold.split(y=y, groups=ep_ind)):
        db.close()
        acc = process_run(_one_fold,
                          args=(train, test, subj_ind, ep_ind, y, db),
                          kwargs=dict(shuffle=shuffle, label_encoder=label_encoder,
                                      classifier_type=classifier_type, epochs=epochs,
                                      batch_size=batch_size,
                                      validation_split=validation_split,
                                      save_classifiers=save_classifiers, i=i,
                                      **classifier_kwargs))
        if save_classifiers:
            acc, saved_clf_name = acc
            saved_clf_names.append(saved_clf_name)
        cross_acc.append(acc)

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
    all_subj = db.get_subject_group()

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
        fast_load=True, n_subjects='all'
):
    if classifier_kwargs is None:
        classifier_kwargs = {}

    db_file = Path(db_file)
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
                    base_dir=base_dir, fast_load=fast_load,
                    n_subjects=n_subjects)

    make_within_subject_classification(db_file, classifier_type,
                                       classifier_kwargs=classifier_kwargs,
                                       n_splits=n_splits, res_handler=results,
                                       save_res=save_res)


def make_cross_subject_classification(db_filename, classifier_type,
                                      leave_out_n_subjects=10, res_handler=None,
                                      save_res=True, epochs=None, batch_size=32,
                                      validation_split=.0,
                                      **classifier_kwargs):
    db = HDF5Dataset(db_filename)
    all_subj = db.get_subject_group()
    y = db.get_y()
    label_encoder = LabelEncoder().fit(y)
    y = label_encoder.transform(y)

    for train_ind, test_ind in LeavePSubjectGroupsOutSequentially(leave_out_n_subjects).split(groups=all_subj):
        clf = init_classifier(classifier_type, db.get_data(train_ind[0]).shape,
                              len(label_encoder.classes_), **classifier_kwargs)

        if epochs is None:
            x_train = db.get_data(train_ind)
            y_train = y[train_ind]
            clf.fit(x_train, y_train)
        else:
            if validation_split > 0:
                # # subject level
                # tr, val = _get_train_val_ind(validation_split, all_subj[train_ind])

                # epoch level
                tr, val = _get_balanced_train_val_ind(validation_split, y[train_ind],
                                                      db.get_epoch_group()[train_ind])

                train_tf_ds = get_tf_dataset(db, y, all_subj[train_ind[tr]]).batch(batch_size)
                train_tf_ds = train_tf_ds.prefetch(tf_data.experimental.AUTOTUNE)
                val_tf_ds = get_tf_dataset(db, y, all_subj[train_ind[val]]).batch(batch_size)
                val_tf_ds = val_tf_ds.cache()
                clf.fit(train_tf_ds, epochs=epochs, validation_data=val_tf_ds)
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
        n_subjects='all'
):
    if classifier_kwargs is None:
        classifier_kwargs = {}

    db_file = Path(db_file)
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
                    base_dir=base_dir, fast_load=fast_load,
                    n_subjects=n_subjects)

    make_cross_subject_classification(db_file, classifier_type,
                                      leave_out_n_subjects=leave_out_n_subjects,
                                      res_handler=results,
                                      save_res=save_res,
                                      **classifier_kwargs)

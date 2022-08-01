from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from tensorflow import data as tf_data

from .ai.classifier import test_classifier, init_classifier, ClassifierType
from .ai.interface import TFBaseNet
from .databases import Databases
from .feature_extraction import FeatureType
from .handlers import ResultHandler, HDF5Dataset
from .handlers.tf import get_tf_dataset
from .model_selection import BalancedKFold, LeavePSubjectGroupsOutSequentially
from .preprocess import generate_db
from .preprocess.io import SubjectHandle
from .utils import mask_to_ind, save_pickle_data, save_to_json
from .validations import validate_feature_classifier_pair


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


def train_test_subject_data(db, subj_ind, classifier_type,
                            *, n_splits=5, shuffle=False,
                            epochs=None, save_classifiers=False,
                            label_encoder=None, batch_size=32,
                            validation_split=.0, patience=15,
                            verbose='auto', weight_file=None,
                            **classifier_kwargs):
    kfold = BalancedKFold(n_splits=n_splits, shuffle=shuffle)

    y = db.get_y()
    ep_ind = db.get_epoch_group()[subj_ind]
    if label_encoder is None:
        label_encoder = LabelEncoder().fit(y)
    y = label_encoder.transform(y[subj_ind])

    cross_acc = list()
    saved_clf_names = list()
    for i, (train_ind, test_ind) in enumerate(kfold.split(y=y, groups=ep_ind)):
        if shuffle:
            np.random.shuffle(train_ind)

        orig_test_mask = db.get_orig_mask()[subj_ind][test_ind]
        orig_test_ind = np.sort(test_ind[orig_test_mask])
        y_train = y[train_ind]
        y_test = y[orig_test_ind]

        clf = init_classifier(classifier_type, db.get_data(subj_ind[0]).shape, len(label_encoder.classes_),
                              **classifier_kwargs)

        if isinstance(clf, TFBaseNet) and weight_file is not None:
            clf.load_weights(weight_file)

        if epochs is None:
            x = db.get_data(subj_ind)
            x_train = x[train_ind]
            x_test = x[orig_test_ind]

            # required for sklearn classifiers with n_job > 1, because joblib
            # does not terminate processes and somehow hdf5 db handler is copied
            # to processes which cause later an error, when db should be deleted...
            # Issue: https://github.com/joblib/joblib/issues/945
            db.close()

            clf.fit(x_train, y_train)
        else:
            y_all = label_encoder.transform(db.get_y())
            if validation_split > 0:
                tr, val = _get_balanced_train_val_ind(validation_split, y_train, ep_ind[train_ind])

                orig_val_mask = db.get_orig_mask()[subj_ind][train_ind[val]]
                orig_val_ind = subj_ind[train_ind[val][orig_val_mask]]

                train_tf_ds = get_tf_dataset(db, y_all, subj_ind[train_ind[tr]]).batch(batch_size)
                train_tf_ds = train_tf_ds.prefetch(tf_data.experimental.AUTOTUNE)
                val_tf_ds = get_tf_dataset(db, y_all, orig_val_ind).batch(batch_size)
                val_tf_ds = val_tf_ds.cache()
                clf.fit(train_tf_ds, epochs=epochs, validation_data=val_tf_ds,
                        patience=patience, verbose=verbose)
            else:
                tf_dataset = get_tf_dataset(db, y_all, subj_ind[train_ind]).batch(batch_size)
                tf_dataset = tf_dataset.prefetch(tf_data.experimental.AUTOTUNE)
                clf.fit(tf_dataset, epochs=epochs, verbose=verbose)
            x_test = db.get_data(subj_ind[orig_test_ind])
            clf.evaluate(x_test, y_test)

        acc = test_classifier(clf, x_test, y_test, label_encoder)
        cross_acc.append(acc)

        if save_classifiers:
            base_dir = db.filename.parent
            if isinstance(clf, TFBaseNet):
                file = base_dir.joinpath('tensorflow', f'clf{i}.h5')
                file.parent.mkdir(parents=True, exist_ok=True)
                clf.save(file)
            elif isinstance(clf, (BaseEstimator, Pipeline, ClassifierMixin)):
                file = base_dir.joinpath('sklearn', f'clf{i}.pkl')
                file.parent.mkdir(parents=True, exist_ok=True)
                save_pickle_data(file, clf)
            else:
                raise NotImplementedError()
            saved_clf_names.append(file)

    print(f"Accuracy scores for k-fold crossvalidation: {cross_acc}\n")
    print(f"Avg accuracy: {np.mean(cross_acc):.4f}   +/- {np.std(cross_acc):.4f}")
    if save_classifiers:
        return cross_acc, saved_clf_names
    return cross_acc


def _get_subject_list(subjects, all_subj):
    subject_list = np.unique(all_subj)
    if subjects == 'all':
        pass
    elif isinstance(subjects, int):
        subject_list = subject_list[:subjects]
    else:
        subject_list = subjects
    return subject_list


def make_within_subject_classification(subjects, db_filename, classifier_type, classifier_kwargs=None,
                                       n_splits=5, res_handler=None, save_res=True, hpc_check_point=None):
    if classifier_kwargs is None:
        classifier_kwargs = {}

    db = HDF5Dataset(db_filename)
    all_subj = db.get_subject_group()

    for subj in _get_subject_list(subjects, all_subj):
        print(f'Subject{subj}')
        s_mask = subj == all_subj
        if not any(s_mask):
            print(f'Subject{subj} is not in processed database')
            continue

        subj_ind = mask_to_ind(s_mask)
        cross_acc = train_test_subject_data(db, subj_ind, classifier_type, n_splits=n_splits,
                                            shuffle=True, **classifier_kwargs)

        if res_handler is not None:
            res_handler.add({'Subject': [f'Subject{subj}'],
                             'Accuracy list': [cross_acc],
                             'Std of Avg. Acc': [np.std(cross_acc)],
                             'Avg. Acc': [np.mean(cross_acc)]})
            if save_res:
                res_handler.save()

        if isinstance(hpc_check_point, dict):
            from bionic_apps.external_connections.hpc.utils import PROCESSED_SUBJ
            hpc_check_point[PROCESSED_SUBJ] = int(subj)
            save_to_json(hpc_check_point['filename'], hpc_check_point)

    db.close()
    if res_handler is not None:
        res_handler.print_db_res()


def test_db_within_subject(
        db_name=Databases.PHYSIONET,
        feature_type=FeatureType.HUGINES,
        epoch_tmin=0, epoch_tmax=4,
        window_len=2, window_step=.1, *,
        ch_selection=None,
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
        db_file='tmp/database.hdf5', log_file='tmp/out.csv', base_dir='.',
        save_res=True,
        fast_load=True, subjects='all',
        augment_data=False,
        hpc_check_point=None
):
    if classifier_kwargs is None:
        classifier_kwargs = {}

    db_file = Path(db_file)
    feature_type, classifier_type = validate_feature_classifier_pair(feature_type, classifier_type)

    fix_params = dict(window_len=window_len, window_step=window_step,
                      n_splits=n_splits,
                      ch_mode=ch_selection,
                      classifier=classifier_type.name,
                      augment_data=augment_data)
    fix_params.update(classifier_kwargs)

    results = ResultHandler(fix_params, ['Subject', 'Accuracy list', 'Std of Avg. Acc', 'Avg. Acc'],
                            to_beginning=('Subject',), filename=log_file)

    generate_db(db_name, db_file, feature_type,
                epoch_tmin, epoch_tmax,
                window_len, window_step,
                ch_selection=ch_selection,
                feature_kwargs=feature_kwargs,
                use_drop_subject_list=use_drop_subject_list,
                filter_params=filter_params,
                do_artefact_rejection=do_artefact_rejection,
                balance_data=balance_data,
                subject_handle=subject_handle,
                base_dir=base_dir, fast_load=fast_load,
                subjects=subjects,
                augment_data=augment_data)

    make_within_subject_classification(subjects, db_file, classifier_type,
                                       classifier_kwargs=classifier_kwargs,
                                       n_splits=n_splits, res_handler=results,
                                       save_res=save_res, hpc_check_point=hpc_check_point)


def make_cross_subject_classification(db_filename, classifier_type,
                                      leave_out_n_subjects=10, shuffle=True,
                                      res_handler=None,
                                      save_res=True, epochs=None, batch_size=32,
                                      validation_split=.0, patience=15,
                                      verbose='auto', finetune=False,
                                      finetune_split=5,
                                      **classifier_kwargs):
    db = HDF5Dataset(db_filename)
    all_subj = db.get_subject_group()
    y = db.get_y()
    label_encoder = LabelEncoder().fit(y)
    y = label_encoder.transform(y)

    for train_ind, test_ind in LeavePSubjectGroupsOutSequentially(leave_out_n_subjects).split(groups=all_subj):
        print(f'Training on subjects: {np.unique(all_subj[train_ind])}')
        if shuffle:
            np.random.shuffle(train_ind)
        db.close()

        clf = init_classifier(classifier_type, db.get_data(train_ind[0]).shape,
                              len(label_encoder.classes_), **classifier_kwargs)

        all_subj = db.get_subject_group()

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

                orig_val_mask = db.get_orig_mask()[train_ind[val]]
                orig_val_ind = train_ind[val][orig_val_mask]

                train_tf_ds = get_tf_dataset(db, y, train_ind[tr]).batch(batch_size)
                train_tf_ds = train_tf_ds.prefetch(tf_data.experimental.AUTOTUNE)
                val_tf_ds = get_tf_dataset(db, y, orig_val_ind).batch(batch_size)
                val_tf_ds = val_tf_ds.cache()
                clf.fit(train_tf_ds, epochs=epochs, validation_data=val_tf_ds,
                        patience=patience, verbose=verbose)
            else:
                tf_dataset = get_tf_dataset(db, y, train_ind).batch(batch_size)
                tf_dataset = tf_dataset.prefetch(tf_data.experimental.AUTOTUNE)
                clf.fit(tf_dataset, epochs=epochs, verbose=verbose)

        if isinstance(clf, TFBaseNet) and finetune:
            cp_file = clf.save_weights()
        else:
            cp_file = None
            finetune = False

        # test subjects individually - check network generalization capability
        for subj in np.unique(all_subj[test_ind]):
            print(f'Subject{subj}')
            test_subj_ind = mask_to_ind(subj == all_subj)

            if not finetune:
                orig_test_mask = db.get_orig_mask()[test_subj_ind]
                orig_test_ind = test_subj_ind[orig_test_mask]

                x_test = db.get_data(orig_test_ind)
                y_test = y[orig_test_ind]
                acc = test_classifier(clf, x_test, y_test, label_encoder)

                if res_handler is not None:
                    res_handler.add({'Subject': [f'Subject{subj}'],
                                     'Left out subjects': [np.unique(all_subj[test_ind])],
                                     'Accuracy': [acc]})

            else:
                cross_acc = train_test_subject_data(db, test_subj_ind, classifier_type,
                                                    n_splits=finetune_split, shuffle=shuffle,
                                                    epochs=epochs, label_encoder=label_encoder,
                                                    batch_size=batch_size,
                                                    validation_split=validation_split,
                                                    patience=patience, verbose=verbose,
                                                    weight_file=cp_file, **classifier_kwargs)
                if res_handler is not None:
                    res_handler.add({'Subject': [f'Subject{subj}'],
                                     'Left out subjects': [np.unique(all_subj[test_ind])],
                                     'Accuracy list': [cross_acc],
                                     'Std of Avg. Acc': [np.std(cross_acc)],
                                     'Avg. Acc': [np.mean(cross_acc)]})

            if save_res and res_handler is not None:
                res_handler.save()

    db.close()
    if res_handler is not None:
        if finetune:
            res_handler.print_db_res()
        else:
            res_handler.print_db_res(col='Accuracy')


def test_db_cross_subject(
        db_name=Databases.PHYSIONET,
        feature_type=FeatureType.RAW,
        epoch_tmin=0, epoch_tmax=4,
        window_len=2, window_step=.1, *,
        ch_selection=None,
        feature_kwargs=None,
        use_drop_subject_list=True,
        filter_params=None,
        do_artefact_rejection=True,
        balance_data=True,
        subject_handle=SubjectHandle.INDEPENDENT_DAYS,
        leave_out_n_subjects=10,
        classifier_type=ClassifierType.EEG_NET,
        classifier_kwargs=None,
        finetune=False,
        finetune_split=5,
        db_file='tmp/database.hdf5', log_file='tmp/out.csv', base_dir='.',
        save_res=True,
        fast_load=True,
        subjects='all',
        augment_data=False
):
    if classifier_kwargs is None:
        classifier_kwargs = {}

    db_file = Path(db_file)
    feature_type, classifier_type = validate_feature_classifier_pair(feature_type, classifier_type)

    fix_params = dict(window_len=window_len, window_step=window_step,
                      ch_mode=ch_selection,
                      classifier=classifier_type.name,
                      augment_data=augment_data)
    fix_params.update(classifier_kwargs)

    if finetune:
        results = ResultHandler(fix_params,
                                ['Subject', 'Left out subjects', 'Accuracy list', 'Std of Avg. Acc', 'Avg. Acc'],
                                to_beginning=('Subject',), filename=log_file)
    else:
        results = ResultHandler(fix_params,
                                ['Subject', 'Left out subjects', 'Accuracy'],
                                to_beginning=('Subject',), filename=log_file)

    generate_db(db_name, db_file, feature_type,
                epoch_tmin, epoch_tmax,
                window_len, window_step,
                ch_selection=ch_selection,
                feature_kwargs=feature_kwargs,
                use_drop_subject_list=use_drop_subject_list,
                filter_params=filter_params,
                do_artefact_rejection=do_artefact_rejection,
                balance_data=balance_data,
                subject_handle=subject_handle,
                base_dir=base_dir, fast_load=fast_load,
                subjects=subjects,
                augment_data=augment_data)

    make_cross_subject_classification(db_file, classifier_type,
                                      leave_out_n_subjects=leave_out_n_subjects,
                                      res_handler=results,
                                      save_res=save_res,
                                      finetune=finetune,
                                      finetune_split=finetune_split,
                                      **classifier_kwargs)

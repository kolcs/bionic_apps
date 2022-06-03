import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, NuSVC

from .feature_extraction import FeatureType
from .handlers import init_hdf5_db, ResultHandler
from .model_selection import BalancedKFold
from .preprocess import generate_eeg_db
from .utils import init_base_config


def get_ensemble_clf(mode='ensemble'):
    level0 = [
        ('SVM', SVC(C=15, gamma=.01, cache_size=512, probability=True)),
        ('nuSVM', NuSVC(nu=.32, gamma=.015, cache_size=512, probability=True)),
        ('Extra Tree', ExtraTreesClassifier(n_estimators=500, criterion='gini')),
        ('Random Forest', RandomForestClassifier(n_estimators=500, criterion='gini')),
        ('Naive Bayes', GaussianNB()),
        ('KNN', KNeighborsClassifier())
    ]

    if mode == 'ensemble':
        level1 = SVC()
        final_clf = StackingClassifier(level0, level1, n_jobs=len(level0))
    elif mode == 'voting':
        final_clf = VotingClassifier(level0, voting='soft', n_jobs=len(level0))
    else:
        raise ValueError(f'Mode {mode} is not an ensemble mode.')

    clf = make_pipeline(
        # PCA(n_components=.97),
        StandardScaler(),
        final_clf
    )
    return clf


def test_classifier(clf, x_test, y_test, le):
    y_pred = clf.predict(x_test)
    y_pred = le.inverse_transform(y_pred)
    y_test = le.inverse_transform(y_test)

    # https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-and-f-measures
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print(class_report)
    print(f"Confusion matrix:\n{conf_matrix}\n")
    print(f"Accuracy score: {acc}\n")
    return acc


def train_test_data(classifier, x, y, groups, lab_enc, *, n_splits=5, shuffle=False, classifier_kwargs=None):
    if classifier_kwargs is None:
        classifier_kwargs = {}
    else:
        classifier_kwargs = classifier_kwargs.copy()

    try:
        epochs = classifier_kwargs.pop('epochs')
    except KeyError:
        epochs = None

    # kfold = StratifiedGroupKFold if groups is not None else StratifiedKFold
    # kfold = kfold(n_splits=3, shuffle=True)

    kfold = BalancedKFold(n_splits=n_splits, shuffle=shuffle)
    cross_acc = list()
    for train, test in kfold.split(y=y, groups=groups):
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]

        if isinstance(classifier, str):
            if classifier == 'eegnet':
                from ai import init_classifier, ClassifierType
                clf = init_classifier(ClassifierType.EEG_NET, x[0].shape, len(lab_enc.classes_))
            elif classifier == 'ensemble':
                clf = get_ensemble_clf()
            elif classifier == 'voting':
                clf = get_ensemble_clf(mode='voting')
            else:
                raise NotImplementedError(f'{classifier} is not implemented')
        elif issubclass(classifier, ClassifierMixin):
            clf = make_pipeline(
                # PCA(n_components=.97),
                StandardScaler(),
                classifier(**classifier_kwargs)
            )
        else:
            raise TypeError(f'Object {type(classifier)} is not a classifier.')

        if epochs is None:
            clf.fit(x_train, y_train)
        else:
            clf.fit(x_train, y_train, epochs=epochs)

        acc = test_classifier(clf, x_test, y_test, lab_enc)
        cross_acc.append(acc)

    print(f"Accuracy scores for k-fold crossvalidation: {cross_acc}\n")
    print(f"Avg accuracy: {np.mean(cross_acc):.4f}   +/- {np.std(cross_acc):.4f}")
    return cross_acc


def make_within_subject_classification(db_filename, classifier, classifier_kwargs=None,
                                       n_splits=5, res_handler=None):
    db, y_all, subj_ind, ep_ind, le = init_hdf5_db(db_filename)

    for subj in np.unique(subj_ind):
        print(f'Subject{subj}')

        x = db.get_data(subj_ind == subj)
        y = y_all[subj_ind == subj]
        groups = ep_ind[subj_ind == subj]

        cross_acc = train_test_data(classifier, x, y, groups=groups, lab_enc=le,
                                    n_splits=n_splits, shuffle=True, classifier_kwargs=classifier_kwargs)

        if res_handler is not None:
            res_handler.add({'Subject': [f'Subject{subj}'],
                             'Accuracy list': [cross_acc],
                             'Std of Avg. Acc': [np.std(cross_acc)],
                             'Avg. Acc': [np.mean(cross_acc)]})
            res_handler.save()
    db.close()


def test_eegdb_within_subject(f_type=FeatureType.HUGINES,
                              window_len=.05, window_step=.01, *,
                              n_splits=5,
                              classifier='ensemble',
                              classifier_kwargs=None,
                              ch_mode='all', ep_mode='distinct',
                              db_file='database.hdf5', log_file='out.csv', base_dir='.'):
    if classifier_kwargs is None:
        classifier_kwargs = {}

    if classifier == 'eegnet':
        f_type = FeatureType.RAW

    fix_params = dict(window_len=window_len, window_step=window_step,
                      n_splits=n_splits,
                      ch_mode=ch_mode, ep_mode=ep_mode,
                      classifier=classifier if isinstance(classifier, str) else classifier.__name__)
    fix_params.update(classifier_kwargs)

    results = ResultHandler(fix_params, ['Subject', 'Accuracy list', 'Std of Avg. Acc', 'Avg. Acc'],
                            to_beginning=('Subject',), filename=log_file)

    generate_eeg_db(db_file, f_type, init_base_config(base_dir), window_len, window_step)

    make_subject_test(db_file, classifier, classifier_kwargs=classifier_kwargs,
                      use_ep_groups=window_step < window_len,
                      n_splits=n_splits,
                      res_handler=results)

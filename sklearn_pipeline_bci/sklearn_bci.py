from enum import Enum, auto

import mne
import numpy as np
import pandas as pd
import pywt
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocess import DataLoader, get_epochs_from_raw


def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def wavelet_denoising(x, wavelet='db2', level=3):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')


def get_data(epochs):
    """Method which extract data and label from epochs"""
    # Convert data from volt to microvolt
    # Pytorch expects float32 for input and int64 for labels.
    data = epochs.get_data() * 1e6
    y = epochs.events[:, 2].astype(np.int64)  # 2,3 -> 0,1,2->> to set the values to zero one and two
    return data, y


def windowed_view(data, window_length, window_step):
    """Windower method which windows a given signal in to a given window size"""
    from numpy.lib.stride_tricks import as_strided

    overlap = window_length - window_step
    new_shape = data.shape[:-1] + ((data.shape[-1] - overlap) // window_step,
                                   window_length)
    new_strides = (data.strides[:-1] + (window_step * data.strides[-1],) +
                   data.strides[-1:])

    result = as_strided(data, shape=new_shape, strides=new_strides)

    return np.transpose(result, (1, 0, 2))


def windower(X, window_length, window_step, fs):
    """Window holder calling windower method"""
    # norm = np.linalg.norm(X)
    # X = X / norm
    windowed_tasks = []
    for i in range(len(X)):
        x = windowed_view(X[i, :, :], int(window_length * fs), int(window_step * fs))
        windowed_tasks.append(np.array(x))
    return np.array(windowed_tasks)


def time_features_estimation(data):
    """
    Compute time features from signal using sliding window method.
    """
    frame = data.shape[-1]
    feature = []

    th = np.mean(data, axis=-1) + 3 * np.std(data, axis=-1)
    abs_dif = np.abs(np.diff(data))

    feature.append(np.var(data, axis=-1))  # var
    feature.append(np.sqrt(np.mean(data ** 2, axis=-1)))  # rms
    feature.append(np.sum(np.abs(data), axis=-1))  # Integral
    feature.append(np.sum(np.abs(data), axis=-1) / frame)  # Mean Absolute Value
    feature.append(np.exp(np.sum(np.log10(np.abs(data)), axis=-1) / frame))
    feature.append(np.sum(abs_dif, axis=-1))  # Wavelength
    feature.append(np.sum(abs_dif, axis=-1) / frame)  # Average Amplitude Change
    feature.append(np.sqrt(
        (1 / (frame - 1)) * np.sum(np.diff(data) ** 2, axis=-1)))  # Difference absolute standard deviation value
    feature.append(np.sum(np.where(np.diff(np.sign(data)) != 0, 1, 0), axis=-1))  # Zero-Crossing
    th1 = np.array([[t] * abs_dif.shape[-1] for t in th])
    feature.append(np.sum(np.where(abs_dif >= th1, 1, 0), axis=-1))  # Wilson amplitude
    th2 = np.array([[t] * data.shape[-1] for t in th])
    feature.append(np.sum(np.where(data >= th2, 1, 0), axis=-1) / frame)  # Opulse percentage rate

    return np.hstack(feature)


class FeatureExtraction(Enum):
    SIMPLE_TIME1 = auto()
    SIMPLE_TIME2 = auto()
    SIMPLE_TIME3 = auto()


def extract_feature(windowed_task, label, feature_type=FeatureExtraction.SIMPLE_TIME1):
    eps, wins, chs, tdata = np.shape(windowed_task)
    epoch_holder = []
    for ep in range(eps):
        window_holder = []
        for w in range(wins):
            window = []
            data = windowed_task[ep, w, :, :]

            if feature_type is FeatureExtraction.SIMPLE_TIME1:
                window.append((np.sum(data, axis=-1) ** 2 + np.sum(data, axis=-1)) / 2)
                window.append((np.mean(data, axis=-1) ** 2 + np.mean(data, axis=-1)) / 2)
                window.append((np.var(data, axis=-1) ** 2 + np.var(data, axis=-1)) / 2)
                window.append((np.std(data, axis=-1) ** 2 + np.std(data, axis=-1)) / 2)
                window.append(np.max(data, axis=-1))
                window.append(np.min(data, axis=-1))
                window.append(np.percentile(data, 25, axis=-1))
                window.append(np.percentile(data, 50, axis=-1))
                window.append(np.percentile(data, 75, axis=-1))
                hist_count, hist_val = np.histogram(data)
                window.append(hist_count)
                window.append(hist_val)
                window.append(np.quantile(data, q=0.25, axis=-1))
                window.append(np.ptp(data, axis=-1))

            elif feature_type is FeatureExtraction.SIMPLE_TIME2:
                # window.append(np.sum(data,axis=-1))
                window.append(np.std(data, axis=-1))
                # window.append(np.var(data,axis=-1))
                # window.append(np.mean(data,axis=-1))
                window.append(np.sqrt(np.mean(data ** 2, axis=-1)))  # rms
                window.append(np.sum(np.abs(data), axis=-1))  # integral
                window.append(np.sum(np.abs(np.diff(data, axis=-1)), axis=-1))  # wavelenght

            elif feature_type is FeatureExtraction.SIMPLE_TIME3:
                window.append(time_features_estimation(data))

            else:
                raise NotImplementedError(f'{feature_type.name} is not implemented.')

            window_holder.append(np.hstack(window))
        x = np.vstack(window_holder)
        r = [label[ep] for r in range(x.shape[0])]
        y = np.insert(x, x.shape[1], r, axis=1)
        # print("epoch extracted", ep)
        # print(y.shape)
        epoch_holder.append(y)

    return np.array(epoch_holder)


# get a list of models to evaluate
def get_models(n_jobs=None):
    models = dict()
    # models['lr'] = LogisticRegression()
    models['KNN'] = KNeighborsClassifier()
    models['Decision Tree'] = DecisionTreeClassifier()
    models['SVM'] = SVC(cache_size=512)
    models['Naive Bayes'] = GaussianNB()
    models['Random Forest'] = RandomForestClassifier()
    models['Extra Tree'] = ExtraTreesClassifier()
    # models['AB']= AdaBoostClassifier(n_estimators=5000, random_state=seed)
    # models['GBC']= GradientBoostingClassifier(n_estimators=5000, random_state=seed)
    models['Ensemble'] = get_stacking(n_jobs)
    return models


def get_stacking(n_jobs=None):
    level0 = list()

    # level0.append(('Logistic Regression', LogisticRegression()))
    # level0.append(('KNN', KNeighborsClassifier()))
    level0.append(('SVM', SVC(cache_size=512)))
    level0.append(('Random Forrest', RandomForestClassifier()))
    level0.append(('Extra Tree', ExtraTreesClassifier()))
    # level0.append(('Decision Tree', DecisionTreeClassifier()))
    level0.append(('Naive Bayes', GaussianNB()))

    # level1 = ExtraTreesClassifier()
    # level1 = LogisticRegression()
    level1 = SVC()

    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5,
                               n_jobs=n_jobs)
    return model


"""Features"""


def var(x):
    return np.var(x, axis=-1)


def rms(x):
    return np.sqrt(np.mean(x ** 2, axis=-1))


def integral(x):
    return np.sum(np.abs(x), axis=-1)


def mav(x):
    return integral(x) / x.shape[-1]


def thing(x):
    return np.exp(np.sum(np.log10(np.abs(x)), axis=-1) / x.shape[-1])


def wavelength(x):
    return np.sum(np.abs(np.diff(x)), axis=-1)


def avg_amp_change(x):
    return wavelength(x) / x.shape[-1]


def diff_abs_std_val(x):
    return np.sqrt(
        (1 / (x.shape[-1] - 1)) * np.sum(np.diff(x) ** 2, axis=-1))


def zero_rossing(x):
    return np.sum(np.where(np.diff(np.sign(x)) != 0, 1, 0), axis=-1)


def wilson_amplitude(x):
    th = np.mean(x, axis=-1) + 3 * np.std(x, axis=-1)
    abs_dif = np.abs(np.diff(x))
    th1 = np.array([[t] * abs_dif.shape[-1] for t in th])
    th1 = np.transpose(th1, (0, 2, 1))
    return np.sum(np.where(abs_dif >= th1, 1, 0), axis=-1)


def opulse_percentage(x):
    th = np.mean(x, axis=-1) + 3 * np.std(x, axis=-1)
    th2 = np.array([[t] * x.shape[-1] for t in th])
    th2 = np.transpose(th2, (0, 2, 1))
    return np.sum(np.where(x >= th2, 1, 0), axis=-1) / x.shape[-1]


def get_complete_steps():
    features = [
        ('var', FunctionTransformer(var)),
        ('rms', FunctionTransformer(rms)),
        ('integral', FunctionTransformer(integral)),
        ('mean abs val', FunctionTransformer(mav)),
        # ('?', FunctionTransformer(thing)),
        ('wavelength', FunctionTransformer(wavelength)),
        ('avg_amp_change', FunctionTransformer(avg_amp_change)),
        ('dif_abs_std', FunctionTransformer(diff_abs_std_val)),
        ('zc', FunctionTransformer(zero_rossing)),
        ('wilson', FunctionTransformer(wilson_amplitude)),
        ('opulse', FunctionTransformer(opulse_percentage))
    ]

    union = FeatureUnion([(name, Pipeline([
        (name, fun),
        ('std scale', StandardScaler()),
        # ('pca', PCA(.9))
    ])) for name, fun in features])

    return union


# evaluate a given model using cross-validation
def evaluate_model(model, x, y, groups):
    steps = [
        ('feature extr', get_complete_steps()),
        # ('pca', PCA(.95)),
        # ('standard scale', StandardScaler()),
        # ('l2 norm', Normalizer()),
        ('model', model)
    ]
    pipeline = Pipeline(steps)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True)
    scores = cross_val_score(pipeline, x, y, groups=groups, scoring='accuracy', cv=cv)
    return np.array(scores)


def test_models(x, y, groups=None, n_jobs=None):
    # get the models to evaluate
    models = get_models(n_jobs)
    # evaluate the models and store results
    results = list()
    for name, model in models.items():
        scores = evaluate_model(model, x, y, groups)
        results.append(scores)
        print('%s model> %.3f (%.5f)' % (name, np.mean(scores), np.std(scores)))
    return np.array(results)


def make_one_subj_test(subj, *, files, task_dict, event_id,
                       window_length=2, window_step=1,
                       n_jobs=None):
    print(f'\nSubject{subj}')
    raw = mne.io.concatenate_raws([mne.io.read_raw(file) for file in files])
    raw.load_data()
    raw.filter(.5, 40)

    epochs = get_epochs_from_raw(raw, task_dict,
                                 epoch_tmin=0, epoch_tmax=4,
                                 event_id=event_id)
    data, label = get_data(epochs)
    # data = wavelet_denoising(data)

    # window the epochs
    windowed_data = windower(data, window_length, window_step, fs=raw.info['sfreq'])

    # feature extractor
    # features = extract_feature(windowed_data, label, FeatureExtraction.SIMPLE_TIME3)
    # X = np.vstack(features[:, :, :-1])
    # Y = features[:, :, -1].ravel()
    features = windowed_data
    X = np.vstack(windowed_data)
    Y = np.hstack(np.array([[l] * windowed_data.shape[1] for l in label]))

    groups = [i // features.shape[1] for i in range(features.shape[0] * features.shape[1])]

    res = test_models(X, Y, groups, n_jobs=n_jobs)
    subj = np.array([[subj]] * 5)
    return np.hstack((subj, res.T))


def make_db_test(window_length=2, window_step=1):
    loader = DataLoader('').use_physionet()
    db_res = Parallel(n_jobs=-2)(delayed(make_one_subj_test)
                                 (subj, files=loader.get_filenames_for_subject(subj),
                                  task_dict=loader.get_task_dict(),
                                  event_id=loader.get_event_id(),
                                  window_length=window_length,
                                  window_step=window_step)
                                 for subj in loader.get_subject_list())

    db_res = np.concatenate(db_res, axis=0)
    return pd.DataFrame(db_res, columns=(['Subjects'] + list(get_models())))


def make_one_test(window_length=2, window_step=1, n_jobs=None):
    loader = DataLoader('').use_physionet()
    db_res = list()
    names = list(get_models())
    for subj in loader.get_subject_list():
        files = loader.get_filenames_for_subject(subj)
        task_dict = loader.get_task_dict()
        event_id = loader.get_event_id()
        db_res.append(make_one_subj_test(subj, files=files, task_dict=task_dict,
                                         event_id=event_id,
                                         window_length=window_length,
                                         window_step=window_step,
                                         n_jobs=n_jobs))
        print('\nDatabase classification results:')
        for name, res in zip(names, np.mean(np.concatenate(db_res, axis=0)[:, 1:], axis=0)):
            print(f'{name}: {res}')
        print()
    db_res = np.concatenate(db_res, axis=0)
    return pd.DataFrame(db_res, columns=(['Subjects'] + names))


# windows length, overlap investigation
def make_window_test(out_file_name='window_test.csv'):
    res = list()
    for w_len in np.arange(.5, 4.1, .5):
        for w_step in np.arange(.1, w_len + .1, .1):
            df = make_db_test(w_len, w_step)
            # df = make_one_test(w_len, w_step)
            df.insert(0, 'window step', w_step)
            df.insert(0, 'window length', w_len)
            res.append(df)

            # save res for every iteration
            pd.concat(res).to_csv(out_file_name, sep=';', encoding='utf-8', index=False)


# make gridsearch on SVM
def svm_param_grid_search_one_subj(subj, files, task_dict, event_id):
    print(f'\nSubject{subj}')
    raw = mne.io.concatenate_raws([mne.io.read_raw(file) for file in files])
    raw.load_data()
    raw.filter(.5, 40)

    epochs = get_epochs_from_raw(raw, task_dict,
                                 epoch_tmin=0, epoch_tmax=4,
                                 event_id=event_id)

    data = epochs.get_data() * 1e6
    labels = [list(epochs[i].event_id[0]) for i in range(len(epochs))]
    label_enc = LabelEncoder()
    labels = label_enc.fit(labels)
    windowed_data = windower(data, window_length=2, window_step=.1, fs=raw.info['sfreq'])

    features = windowed_data
    X = np.vstack(windowed_data)
    Y = np.hstack(np.array([[l] * windowed_data.shape[1] for l in labels]))

    groups = [i // features.shape[1] for i in range(features.shape[0] * features.shape[1])]

    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedGroupKFold(n_splits=10, shuffle=True)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, Y, groups=groups)

    print(
        "The best parameters are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_)
    )
    scores = grid.cv_results_["mean_test_score"].reshape(len(C_range), len(gamma_range))

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    class MidpointNormalize(Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(
        scores,
        interpolation="nearest",
        cmap=plt.cm.hot,
        norm=MidpointNormalize(vmin=0.2, midpoint=0.92),
    )
    plt.xlabel("gamma")
    plt.ylabel("C")
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title(f"Validation accuracy for subject{subj}")
    plt.show()


def svm_param_grid_search():
    loader = DataLoader('').use_physionet()
    for subj in loader.get_subject_list():
        files = loader.get_filenames_for_subject(subj)
        task_dict = loader.get_task_dict()
        event_id = loader.get_event_id()
        svm_param_grid_search_one_subj(subj, files, task_dict, event_id)


if __name__ == '__main__':
    make_one_test(window_length=2, window_step=.1)
    # make_window_test()

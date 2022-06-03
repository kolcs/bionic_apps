import numpy as np
from ai.interface import ClassifierInterface
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, \
    PowerTransformer, QuantileTransformer
from sklearn.svm import SVC


class OnlinePipeline(Pipeline):

    def __init__(self, steps, memory=None, verbose=False):
        super().__init__(steps, memory, verbose)
        self._init_fit = True

    def fit(self, X, y=None, **fit_params):
        if self._init_fit:
            super().fit(X, y, **fit_params)
            self._init_fit = False
        else:
            for i, step in enumerate(self.steps):
                name, est = step
                if i < len(self.steps) - 1:
                    X = est.transform(X)
                else:
                    est.partial_fit(X, y)

        return self


def _init_one_svm(svm, pipeline=('norm',), **svm_kargs):
    if svm is None:
        # svm = OnlinePipeline([('norm', Normalizer()), ('svm', SGDClassifier(**svm_kargs))])
        pipe_list = list()
        for el in pipeline:
            if el == 'norm':
                element = (el, Normalizer())
            elif el == 'standard':
                element = (el, StandardScaler())
            elif el == 'minmax':
                element = (el, MinMaxScaler())
            elif el == 'maxabs':
                element = (el, MaxAbsScaler())
            elif el == 'robust':
                element = (el, RobustScaler())
            elif el == 'power':
                element = (el, PowerTransformer())
            elif el == 'quantile':
                element = (el, QuantileTransformer(output_distribution='normal'))
            else:
                raise ValueError(f'{el} is not in SVM pipeline options.')
            pipe_list.append(element)
        pipe_list.append(('svm', SVC(**svm_kargs)))
        svm = Pipeline(pipe_list)
    return svm


def _fit_one_svm(svm, data, label, num):
    svm.fit(data, label)
    return num, svm


class MultiSVM(ClassifierInterface):
    def __init__(self, **svm_kwargs):
        self._svm_kargs = svm_kwargs
        self._svms = dict()

    def _predict(self, i, data):
        svm = self._svms[i]
        return svm.predict(data)

    def fit(self, X, y, **kwargs):
        """

        Parameters
        ----------
        X : numpy.ndarray
            EEG data to be processed. shape: (n_samples, n_svm, n_features)
        y : numpy.ndarray
            labels

        Returns
        -------

        """
        X = np.array(X)
        n_svms = X.shape[1]
        self._svms = {i: _init_one_svm(self._svms.get(i), **self._svm_kargs) for i in range(n_svms)}
        # self._svms = [SVM(*self._svm_args) for _ in range(n_svms)]  # serial: 3 times slower
        # for i in range(len(self._svms)):
        #     self._fit_svm(i, X[:, i, :], y)
        if len(y.shape) == 2 and y.shape[1] == 1:
            y = np.ravel(y)
        if n_svms > 1:
            svms = Parallel(n_jobs=-2)(
                delayed(_fit_one_svm)(self._svms[i], X[:, i, :], y, i) for i in range(n_svms))
        else:
            svms = [_fit_one_svm(self._svms[0], X[:, 0, :], y, 0)]
        self._svms = dict(svms)

    def predict(self, X):
        X = np.array(X)
        votes = [self._predict(i, X[:, i, :]) for i in range(X.shape[1])]
        # votes.extend([self._predict(X.shape[1] + i, X[:, :, i]) for i in range(X.shape[2])])
        # votes = [self._predict(i, X[:, :, i]) for i in range(X.shape[2])]

        # votes = Parallel(n_jobs=-2)(delayed(self._predict)(i, X[:, i, :]) for i in range(len(self._svms)))
        votes = np.array(votes)
        res = list()
        for i in range(votes.shape[1]):  # counting votes
            unique, count = np.unique(votes[:, i], return_counts=True)
            res.append(unique[np.argmax(count)])
        return res


if __name__ == '__main__':
    pass

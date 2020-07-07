import numpy as np
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC


class OnlinePipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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


class MultiSVM(object):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=None):
        self._svm_args = (C, kernel, degree, gamma, coef0, shrinking, probability, tol, cache_size, class_weight,
                          verbose, max_iter, decision_function_shape, random_state)
        self._svms = dict()

    def _fit_one_svm(self, data, label, num):
        svm = Pipeline([('norm', Normalizer()), ('svm', SVC(*self._svm_args))])  # or StandardScaler()
        svm.fit(data, label)
        return num, svm

    # def _fit_svm(self, i, data, label):
    #     self._svms[i].fit(data, label)

    def _predict(self, i, data):
        svm = self._svms[i]
        return svm.predict(data)

    def fit(self, X, y):
        """

        Parameters
        ----------
        X : numpy.array
            EEG data to be processed. shape: (n_samples, n_svm, n_features)
        y : list
            labels

        Returns
        -------

        """
        X = np.array(X)
        n_svms = X.shape[1]
        # self._svms = [SVM(*self._svm_args) for _ in range(n_svms)]  # serial: 3 times slower
        # for i in range(len(self._svms)):
        #     self._fit_svm(i, X[:, i, :], y)
        svms = Parallel(n_jobs=-2)(delayed(self._fit_one_svm)(X[:, i, :], y, i) for i in range(n_svms))
        self._svms = dict(svms)

        # d = len(self._svms)
        # n_svms = X.shape[2]
        # svms = Parallel(n_jobs=-2)(delayed(self._fit_one_svm)(X[:, :, i], y, i) for i in range(n_svms))
        # self._svms.update(svms)

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

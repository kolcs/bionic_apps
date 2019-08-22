import numpy as np
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder, normalize


def _one_hot_encode_example():
    from config import REST, LEFT_HAND, RIGHT_HAND, BOTH_HANDS, BOTH_LEGS

    # generate single label -- All possible options
    task_list = np.array([REST, LEFT_HAND, RIGHT_HAND, BOTH_LEGS, BOTH_HANDS]).reshape((-1, 1))
    print(type(task_list))

    # create label list which should be one hot encoded
    y = [task_list[np.random.randint(len(task_list))] for _ in range(10)]

    # init encoder
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(task_list)

    # one hot encode
    one_hot = enc.transform(y).toarray()

    print(one_hot)
    print(enc.inverse_transform(one_hot))


def _correct_shape(y):
    y = np.array(y)

    if len(np.shape(y)) == 1:
        y = y.reshape((-1, 1))

    return y


def calculate_sample_weight(labels):
    ind_label = set(labels)
    label_dict = {label: len([i for i, x in enumerate(labels) if x == label]) for label in ind_label}
    a = [v for k, v in label_dict.items()]
    s = np.sum(a)
    div = max(s / a)
    for key in label_dict:
        label_dict[key] = s / label_dict[key] / div
    sample_weight = [label_dict[l] for l in labels]
    return sample_weight


class SVM(svm.SVC):

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=None):
        self._enc = OneHotEncoder(handle_unknown='ignore')

        super(SVM, self).__init__(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking,
                                  probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight,
                                  verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape,
                                  random_state=random_state)

    def set_labels(self, labels):
        """ Label settings for one-hot encoder.

        Set labels here, which should be converted to one-hot vectors.

        Parameters
        ----------
        labels : list
            List of individual labels.
        """
        labels = _correct_shape(labels)
        self._enc.fit(labels)

    # def one_hot_encode(self, y):
    #     return self._enc.transform(y).toarray()
    #
    # def one_hot_decode(self, y):
    #     return self._enc.inverse_transform(y)

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        This function overrides the base function: It normalize the given
        data and converts the labels to one-hot encoded vector.
        Then it calls the super fit method.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] | list of sting
            Target vector relative to X. Can be list of list of strings.

        sample_weight : array-like, shape = [n_samples], optional
            Array of weights that are assigned to individual
            samples. If not provided,
            then each sample is given unit weight.
        """
        X_normalized = normalize(X, norm='l2')
        # y = _correct_shape(y)
        # y = self._enc.transform(y).toarray()  # one hot encode
        # print(y, np.shape(X_normalized), np.shape(y))
        super().fit(X_normalized, y, sample_weight=sample_weight)

    def predict(self, X):
        X_normalized = normalize(X, norm='l2')
        y = super().predict(X_normalized)
        # return self._enc.inverse_transform(y)  # one hot decode
        return y


class LinearSVM(svm.LinearSVC):

    def __init__(self, penalty='l2', loss='squared_hinge', dual=True, tol=1e-4,
                 C=1.0, multi_class='ovr', fit_intercept=True,
                 intercept_scaling=1, class_weight=None, verbose=0,
                 random_state=None, max_iter=1000):
        super().__init__(penalty, loss, dual, tol, C, multi_class, fit_intercept, intercept_scaling,
                         class_weight, verbose, random_state, max_iter)

    def set_labels(self, labels):
        pass

    def fit(self, X, y, sample_weight=None):
        X_normalized = normalize(X, norm='l2')
        if sample_weight is None:
            sample_weight = calculate_sample_weight(y)
        elif sample_weight == 'shrink rest':
            sample_weight = [.25 if label == 'rest' else 1 for label in y]
        super().fit(X_normalized, y, sample_weight=sample_weight)

    def predict(self, X):
        X_normalized = normalize(X, norm='l2')
        y = super().predict(X_normalized)
        return y


if __name__ == '__main__':
    _one_hot_encode_example()
    pass

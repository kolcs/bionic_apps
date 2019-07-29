import numpy as np
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder


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
        labels = np.array(labels)

        if len(np.shape(labels)) == 1:
            labels = labels.reshape((-1, 1))

        self._enc.fit(labels)

    # def one_hot_encode(self, y):
    #     return self._enc.transform(y).toarray()
    #
    # def one_hot_decode(self, y):
    #     return self._enc.inverse_transform(y)

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        This function overrides the base function: It converts the labels to one-hot encoded vector.
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
        y = self._enc.transform(y).toarray()  # one hot encode
        super().fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        y = super().predict(X)
        return self._enc.inverse_transform(y)  # one hot decode


if __name__ == '__main__':
    _one_hot_encode_example()
    pass

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
        labels = np.array(labels)

        if len(np.shape(labels)) == 1:
            labels = labels.reshape((-1, 1))

        self._enc.fit(labels)

    def one_hot_encode(self, y):
        return self._enc.transform(y).toarray()

    def one_hot_decode(self, y):
        return self._enc.inverse_transform(y)

    def fit(self, X, y, sample_weight=None):
        """

        Parameters
        ----------
        X:
        y
        sample_weight

        Returns
        -------

        """
        # todo: continue development
        # X : (n_samples, n_features)
        y = self.one_hot_decode(y)
        super().fit(X, y, sample_weight=sample_weight)


if __name__ == '__main__':
    _one_hot_encode_example()
    pass

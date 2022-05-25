import numpy as np
from sklearn.model_selection import KFold, StratifiedGroupKFold, PredefinedSplit


class BalancedKFold:

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def _set_n_split(self, y, groups):
        if self.n_splits == 'auto':
            if groups is not None:
                self.n_splits = min([len(np.unique(groups[label == y])) for label in np.unique(y)])
            else:
                self.n_splits = 5

    def split(self, x=None, y=None, groups=None):
        assert y is not None, 'y must be defined'
        self._set_n_split(y, groups)

        y_uniqe = np.unique(y)
        label_test_folds = {label: [] for label in y_uniqe}
        for label in y_uniqe:
            ind = np.arange(len(y))[y == label]
            lb_group = groups[ind] if groups is not None else None
            kfold = KFold if groups is None else StratifiedGroupKFold
            kfold = kfold(self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            lb_y = lb_group if groups is not None else ind
            for _, test in kfold.split(ind, lb_y, groups=lb_group):
                label_test_folds[label].append(ind[test])

        test_fold = np.zeros_like(y)
        for i in range(self.n_splits):
            if self.shuffle:
                np.random.shuffle(y_uniqe)
            for label in y_uniqe:
                test_fold[label_test_folds[label][i]] = i

        ps = PredefinedSplit(test_fold)
        for train, test in ps.split():
            assert not any(item in train for item in test), 'Splitting error.'
            yield train, test

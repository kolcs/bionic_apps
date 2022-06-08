from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC, NuSVC

from .interface import ClassifierInterface


def _select_fft(x, i):
    return x[:, i, :]


class VotingSVM(ClassifierInterface):

    def __init__(self, norm=StandardScaler):
        self.norm = norm
        self._model = None

    def fit(self, x, y=None, **kwargs):
        n_svms = x.shape[1]
        inner_clfs = [(f'unit{i}', make_pipeline(FunctionTransformer(_select_fft, kw_args={'i': i}),
                                                 self.norm(), SVC(probability=True)))
                      for i in range(n_svms)]
        self._model = VotingClassifier(inner_clfs, voting='soft', n_jobs=len(inner_clfs)) \
            if len(inner_clfs) > 1 else inner_clfs[0][1]
        self._model.fit(x, y)

    def predict(self, x):
        return self._model.predict(x)


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
        level1 = LinearDiscriminantAnalysis()
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

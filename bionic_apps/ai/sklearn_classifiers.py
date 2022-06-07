from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC

from .interface import ClassifierInterface


def select_fft(x, i):
    return x[:, i, :]


class VotingSVM(ClassifierInterface):

    def __init__(self, norm=StandardScaler):
        self.norm = norm
        self._model = None

    def fit(self, x, y=None, **kwargs):
        n_svms = x.shape[1]
        inner_clfs = [(f'unit{i}', make_pipeline(FunctionTransformer(select_fft, kw_args={'i': i}),
                                                 self.norm(), SVC(probability=True)))
                      for i in range(n_svms)]
        self._model = VotingClassifier(inner_clfs, voting='soft', n_jobs=len(inner_clfs)) \
            if len(inner_clfs) > 1 else inner_clfs[0][1]

    def predict(self, x):
        return self._model.predict(x)

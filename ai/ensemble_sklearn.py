from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from ai.interface import ClassifierInterface


class Ensemble(ClassifierInterface):

    def __init__(self):
        self._classifiers = {
            'Logistic Regression': LogisticRegression(solver='lbfgs', max_iter=3000),
            'KNN': KNeighborsClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_features=12, min_samples_split=14)
        }
        self._model = VotingClassifier(
            estimators=[(est_name, est) for est_name, est in self._classifiers.items()],
            voting='soft',
            n_jobs=-1,
        )

    def predict(self, x):
        return self._model.predict(x)

    def fit(self, x, y, **kwargs):
        self._model.fit(x, y)

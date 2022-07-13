from sklearn.ensemble import ExtraTreesClassifier

from bionic_apps.ai import ClassifierType
from bionic_apps.databases import Databases
from bionic_apps.feature_extraction import FeatureType, get_hugines_transfromer
from bionic_apps.offline_analyses import test_db_within_subject

feature_type = FeatureType.USER_PIPELINE
classifier_type = ClassifierType.USER_DEFINED

db_name = Databases.PHYSIONET
fs = 160
chs = 64

filter_params = dict(
    order=5, l_freq=1, h_freq=45
)

feature_kwargs = dict(
    pipeline=get_hugines_transfromer()
)

classifier_kwargs = dict(
    classifier=ExtraTreesClassifier(n_estimators=250, n_jobs=-2)
)

test_db_within_subject(db_name, feature_type,
                       filter_params=filter_params,
                       feature_kwargs=feature_kwargs,
                       classifier_type=classifier_type,
                       classifier_kwargs=classifier_kwargs,
                       do_artefact_rejection=True, fast_load=True,
                       log_file='out.csv')

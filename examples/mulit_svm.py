from bionic_apps.ai import ClassifierType
from bionic_apps.databases import EEG_Databases
from bionic_apps.feature_extraction import eeg_bands
from bionic_apps.offline_analyses import test_eegdb_within_subject

band = eeg_bands['range40'].copy()
feature_type = band.pop('feature_type')
filter_params = dict(  # required for FASTER artefact filter
    order=5, l_freq=1, h_freq=45
)
test_eegdb_within_subject(EEG_Databases.PHYSIONET, feature_type, feature_kwargs=band,
                          classifier_type=ClassifierType.VOTING_SVM,
                          do_artefact_rejection=True, fast_load=True,
                          filter_params=filter_params,
                          log_file='out.csv')

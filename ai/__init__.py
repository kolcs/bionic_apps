from .classifier import ClassifierType
from .svm import MultiSVM


def init_classifier(classifier_type, **kwargs):
    if classifier_type is ClassifierType.SVM:
        classifier = MultiSVM(**kwargs)

    else:
        raise NotImplementedError('Classifier {} is not implemented.'.format(classifier_type.name))
    return classifier

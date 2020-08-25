from .classifier import ClassifierType
from .svm import MultiSVM
from .neural_networks import VGG19


def init_classifier(classifier_type, classes, input_shape, **kwargs):
    if classifier_type is ClassifierType.SVM:
        classifier = MultiSVM(**kwargs)
    elif classifier_type is ClassifierType.VGG19:
        classifier = VGG19(classes, input_shape, **kwargs)

    else:
        raise NotImplementedError('Classifier {} is not implemented.'.format(classifier_type.name))
    return classifier

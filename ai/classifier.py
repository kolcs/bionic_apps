from enum import Enum, auto

from ai.neural_networks import VGG, VggType, Dense, DenseNetType
from ai.svm import MultiSVM


class ClassifierType(Enum):
    SVM = auto()
    DENSE_NET_121 = auto()
    DENSE_NET_169 = auto()
    DENSE_NET_201 = auto()
    VGG16 = auto()
    VGG19 = auto()


def init_classifier(classifier_type, classes, input_shape, **kwargs):
    if classifier_type is ClassifierType.SVM:
        classifier = MultiSVM(**kwargs)
    elif classifier_type is ClassifierType.VGG16:
        classifier = VGG(VggType.VGG16, classes, input_shape, **kwargs)
    elif classifier_type is ClassifierType.VGG19:
        classifier = VGG(VggType.VGG19, classes, input_shape, **kwargs)
    elif classifier_type is ClassifierType.DENSE_NET_121:
        classifier = Dense(DenseNetType.DN121, classes, input_shape, **kwargs)
    elif classifier_type is ClassifierType.DENSE_NET_169:
        classifier = Dense(DenseNetType.DN169, classes, input_shape, **kwargs)
    elif classifier_type is ClassifierType.DENSE_NET_201:
        classifier = Dense(DenseNetType.DN201, classes, input_shape, **kwargs)

    else:
        raise NotImplementedError('Classifier {} is not implemented.'.format(classifier_type.name))
    return classifier

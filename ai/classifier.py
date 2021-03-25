from enum import Enum, auto

from ai.kolcs_neural_networks import VGG, VggType, DenseNet, DenseNetType, CascadeConvRecNet, BasicNet
from ai.svm import MultiSVM


class ClassifierType(Enum):
    SVM = auto()
    DENSE_NET_121 = auto()
    DENSE_NET_169 = auto()
    DENSE_NET_201 = auto()
    VGG16 = auto()
    VGG19 = auto()
    CASCADE_CONV_REC = auto()
    KOLCS_NET = auto()


def init_classifier(classifier_type, input_shape, classes, **kwargs):
    if classifier_type is ClassifierType.SVM:
        classifier = MultiSVM(**kwargs)
    elif classifier_type is ClassifierType.VGG16:
        classifier = VGG(VggType.VGG16, input_shape, classes, **kwargs)
    elif classifier_type is ClassifierType.VGG19:
        classifier = VGG(VggType.VGG19, input_shape, classes, **kwargs)
    elif classifier_type is ClassifierType.DENSE_NET_121:
        classifier = DenseNet(DenseNetType.DN121, input_shape, classes, **kwargs)
    elif classifier_type is ClassifierType.DENSE_NET_169:
        classifier = DenseNet(DenseNetType.DN169, input_shape, classes, **kwargs)
    elif classifier_type is ClassifierType.DENSE_NET_201:
        classifier = DenseNet(DenseNetType.DN201, input_shape, classes, **kwargs)
    elif classifier_type is ClassifierType.CASCADE_CONV_REC:
        classifier = CascadeConvRecNet(input_shape, classes, **kwargs)
    elif classifier_type is ClassifierType.KOLCS_NET:
        classifier = BasicNet(input_shape, classes)

    else:
        raise NotImplementedError('Classifier {} is not implemented.'.format(classifier_type.name))
    return classifier

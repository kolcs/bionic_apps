from enum import Enum, auto

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from .kolcs_neural_networks import VGG, VggType, DenseNet, DenseNetType, CascadeConvRecNet, BasicNet
from .pusar_neural_networks import EEGNet
from .sklearn_classifiers import VotingSVM, get_ensemble_clf


class ClassifierType(Enum):
    USER_DEFINED = auto()

    # sklearn classifiers
    VOTING_SVM = auto()
    ENSEMBLE = auto()
    VOTING = auto()

    # neural networks
    DENSE_NET_121 = auto()
    DENSE_NET_169 = auto()
    DENSE_NET_201 = auto()
    VGG16 = auto()
    VGG19 = auto()
    CASCADE_CONV_REC = auto()
    KOLCS_NET = auto()
    EEG_NET = auto()


def init_classifier(classifier_type, input_shape, classes,
                    *, classifier=None, **kwargs):
    if classifier_type is ClassifierType.VOTING_SVM:
        classifier = VotingSVM(**kwargs)
    elif classifier_type is ClassifierType.ENSEMBLE:
        classifier = get_ensemble_clf()
    elif classifier_type is ClassifierType.VOTING:
        classifier = get_ensemble_clf('voting')
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
    elif classifier_type is ClassifierType.EEG_NET:
        classifier = EEGNet(input_shape, classes, **kwargs)
    elif classifier_type is ClassifierType.USER_DEFINED:
        assert classifier is not None, f'classifier must be defined!'
    else:
        raise NotImplementedError('Classifier {} is not implemented.'.format(classifier_type.name))
    return classifier


def test_classifier(clf, x_test, y_test, le):
    y_pred = clf.predict(x_test)
    y_pred = le.inverse_transform(y_pred)
    y_test = le.inverse_transform(y_test)

    # https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-and-f-measures
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print(class_report)
    print(f"Confusion matrix:\n{conf_matrix}\n")
    print(f"Accuracy score: {acc}\n")
    return acc

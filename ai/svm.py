import numpy as np
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder, normalize
# from sklearn.utils import compute_class_weight
from svmutil import *
import sys
import os
from subprocess import *

from preprocess import make_dir


def _one_hot_encode_example():
    from config import REST, LEFT_HAND, RIGHT_HAND, BOTH_HANDS, BOTH_LEGS

    # generate single label -- All possible options
    task_list = np.array([REST, LEFT_HAND, RIGHT_HAND, BOTH_LEGS, BOTH_HANDS]).reshape((-1, 1))
    print(type(task_list))

    # create label list which should be one hot encoded
    y = [task_list[np.random.randint(len(task_list))] for _ in range(10)]

    # init encoder
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(task_list)

    # one hot encode
    one_hot = enc.transform(y).toarray()

    print(one_hot)
    print(enc.inverse_transform(one_hot))


def _correct_shape(y):
    y = np.array(y)

    if len(np.shape(y)) == 1:
        y = y.reshape((-1, 1))

    return y


# def calculate_sample_weight(labels):
#     ind_label = set(labels)
#     label_dict = {label: len([i for i, x in enumerate(labels) if x == label]) for label in ind_label}
#     a = [v for k, v in label_dict.items()]
#     s = np.sum(a)
#     div = max(s / a)
#     for key in label_dict:
#         label_dict[key] = s / label_dict[key] / div
#     sample_weight = [label_dict[l] for l in labels]
#     return sample_weight


class SVM(svm.SVC):

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=None):
        self._enc = OneHotEncoder(handle_unknown='ignore')

        super(SVM, self).__init__(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking,
                                  probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight,
                                  verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape,
                                  random_state=random_state)

    def set_labels(self, labels):
        """ Label settings for one-hot encoder.

        Set labels here, which should be converted to one-hot vectors.

        Parameters
        ----------
        labels : list
            List of individual labels.
        """
        labels = _correct_shape(labels)
        self._enc.fit(labels)

    # def one_hot_encode(self, y):
    #     return self._enc.transform(y).toarray()
    #
    # def one_hot_decode(self, y):
    #     return self._enc.inverse_transform(y)

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        This function overrides the base function: It normalize the given
        data and converts the labels to one-hot encoded vector.
        Then it calls the super fit method.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] | list of sting
            Target vector relative to X. Can be list of list of strings.

        sample_weight : array-like, shape = [n_samples], optional
            Array of weights that are assigned to individual
            samples. If not provided,
            then each sample is given unit weight.
        """
        X_normalized = normalize(X, norm='l2')
        # y = _correct_shape(y)
        # y = self._enc.transform(y).toarray()  # one hot encode
        # print(y, np.shape(X_normalized), np.shape(y))
        super().fit(X_normalized, y, sample_weight=sample_weight)

    def predict(self, X):
        X_normalized = normalize(X, norm='l2')
        y = super().predict(X_normalized)
        # return self._enc.inverse_transform(y)  # one hot decode
        return y


class LinearSVM(svm.LinearSVC):

    def __init__(self, penalty='l2', loss='squared_hinge', dual=True, tol=1e-4,
                 C=1.0, multi_class='ovr', fit_intercept=True,
                 intercept_scaling=1, class_weight=None, verbose=0,
                 random_state=None, max_iter=1000):
        super().__init__(penalty, loss, dual, tol, C, multi_class, fit_intercept, intercept_scaling,
                         class_weight, verbose, random_state, max_iter)

    def set_labels(self, labels):
        pass

    def fit(self, X, y, sample_weight=None):
        X_normalized = normalize(X, norm='l2')
        # if sample_weight is None:
        #     sample_weight = calculate_sample_weight(y)
        # elif sample_weight == 'shrink rest':
        #     sample_weight = [.25 if label == 'rest' else 1 for label in y]
        super().fit(X_normalized, y, sample_weight=sample_weight)

    def predict(self, X):
        X_normalized = normalize(X, norm='l2')
        y = super().predict(X_normalized)
        return y


class libsvm_SVC(object):

    def __init__(self, C=1, kernel='rbf', degree=3, gamma=None, coef0=0.0, cache_size=200, tol=1e-3, shrinking=True,
                 probability=False, class_weight=None, quiet_mode=True):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.cache_size = cache_size
        self.tol = tol
        self.shrinking = int(shrinking)
        self.probability = int(probability)
        self.class_weight = class_weight
        self.quiet_mode = quiet_mode

        self._labels = None
        self._label_to_int = None
        self._int_to_label = None
        self._model = None

    def set_labels(self, labels):
        self._labels = np.array(labels)
        self._label_to_int = {label: i for i, label in enumerate(self._labels)}
        self._int_to_label = {i: label for i, label in enumerate(self._labels)}

    def _set_svm_options(self, class_weight, gamma):
        params = '-s 0'
        if self.kernel == 'rbf':
            kernel = 2
        elif self.kernel == 'linear':
            kernel = 0
        elif self.kernel == 'poly':
            kernel = 1
        elif self.kernel == 'sigmoid':
            kernel = 3
        else:  # precomputed
            kernel = 4
        params += ' -t {}'.format(kernel)
        params += ' -d {}'.format(self.degree)
        params += ' -g {}'.format(gamma)
        params += ' -r {}'.format(self.coef0)
        params += ' -c {}'.format(self.C)
        params += ' -m {}'.format(self.cache_size)
        params += ' -e {}'.format(self.tol)
        params += ' -h {}'.format(self.shrinking)
        params += ' -b {}'.format(self.probability)
        if class_weight is not None:
            for i, weight in enumerate(class_weight):
                params += ' -w{} {}'.format(i, weight)
        if self.quiet_mode:
            params += ' -q'

        return params

    def _encode_y(self, y):
        conv_y = [self._label_to_int[el] for el in y]
        return conv_y

    def _decode_y(self, y):
        return [self._int_to_label[i] for i in y]

    def _compute_class_weight(self, y):
        class_weight = np.ones(np.shape(self._labels))
        for key in self.class_weight:
            class_weight[self._label_to_int[key]] = self.class_weight[key]
        return class_weight

    def fit(self, X, y):
        X_normalized = normalize(X, norm='l2')
        class_weight = self.class_weight
        gamma = self.gamma
        if self.class_weight is not None:
            class_weight = self._compute_class_weight(y)
        if self.gamma is None:
            gamma = 1 / len(set(y))

        options = self._set_svm_options(class_weight, gamma)
        y = self._encode_y(y)
        self._model = svm_train(y, X_normalized, options)

    def predict(self, X):
        X_normalized = normalize(X, norm='l2')
        options = '-b {}'.format(self.probability)
        if self.quiet_mode:
            options += ' -q'

        pred_labels, _, pred_values = svm_predict([], X_normalized, self._model, options)
        pred_labels = self._decode_y(pred_labels)
        return pred_labels


class libsvm_cuda(object):

    def __init__(self, C=1, degree=3, gamma=None, coef0=0.0, cache_size=200, tol=1e-3, shrinking=True,
                 probability=False, class_weight=None, quiet_mode=True,
                 cuda_libsvm_path=r"D:\Users\Csabi\Downloads\CUDA-libsvm\binaries\windows\x64\windows"):
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.cache_size = cache_size
        self.tol = tol
        self.shrinking = int(shrinking)
        self.probability = int(probability)
        self.class_weight = class_weight
        self.quiet_mode = quiet_mode

        self._labels = None
        self._label_to_int = None
        self._int_to_label = None
        self._model = None

        self._cuda_libsvm_path = cuda_libsvm_path

        if sys.platform == 'win32':
            self._svmscale_exe = os.path.join(cuda_libsvm_path, "svm-scale.exe")
            self._svmtrain_exe = os.path.join(cuda_libsvm_path, "svm-train-gpu.exe")
            self._svmpredict_exe = os.path.join(cuda_libsvm_path, "svm-predict.exe")
        else:
            self._svmscale_exe = os.path.join(cuda_libsvm_path, "svm-scale")
            self._svmtrain_exe = os.path.join(cuda_libsvm_path, "svm-train-gpu")
            self._svmpredict_exe = os.path.join(cuda_libsvm_path, "svm-predict")

        self._check_path_to_cuda_libsvm()

        make_dir('tmp')
        self._train_file = os.path.join('tmp', 'tmp_svm_file')
        self._scaled_file = self._train_file + ".scale"
        self._model_file = self._train_file + ".model"
        self._range_file = self._train_file + ".range"
        self._predict_file = self._train_file + ".predict"

    def _check_path_to_cuda_libsvm(self):
        assert os.path.exists(self._svmscale_exe), "svm-scale executable not found on path: {}".format(
            self._svmscale_exe)
        assert os.path.exists(self._svmtrain_exe), "svm-train-gpu executable not found on path: {}".format(
            self._svmtrain_exe)
        assert os.path.exists(self._svmpredict_exe), "svm-predict executable not found on path: {}".format(
            self._svmpredict_exe)

    def set_labels(self, labels):
        self._labels = np.array(labels)
        self._label_to_int = {label: i for i, label in enumerate(self._labels)}
        self._int_to_label = {i: label for i, label in enumerate(self._labels)}

    def _set_svm_options(self, class_weight, gamma):
        params = '-d {}'.format(self.degree)
        params += ' -g {}'.format(gamma)
        params += ' -r {}'.format(self.coef0)
        params += ' -c {}'.format(self.C)
        params += ' -m {}'.format(self.cache_size)
        params += ' -e {}'.format(self.tol)
        params += ' -h {}'.format(self.shrinking)
        params += ' -b {}'.format(self.probability)
        if class_weight is not None:
            for i, weight in enumerate(class_weight):
                params += ' -w{} {}'.format(i, weight)
        if self.quiet_mode:
            params += ' -q'

        return params

    def _encode_y(self, y):
        conv_y = [self._label_to_int[el] for el in y]
        return conv_y

    def _decode_y(self, y):
        return [self._int_to_label[i] for i in y]

    def _compute_class_weight(self):
        class_weight = np.ones(np.shape(self._labels))
        for key in self.class_weight:
            class_weight[self._label_to_int[key]] = self.class_weight[key]
        return class_weight

    @staticmethod
    def _convert_to_input_file(filename, X, y=None):
        if y is not None:
            assert len(X) == len(y), "Feature and label number is not equal."

        with open(filename, 'w') as file:
            for i, feature in enumerate(X):
                if y is not None:
                    label = y[i]
                else:
                    label = -1
                file.write(str(label))
                for j, f in enumerate(feature):
                    file.write(' {}:{}'.format(j, f))
                file.write('\n')

    def _scale_input_file(self):
        cmd = '{0} -s "{1}" "{2}" > "{3}"'.format(self._svmscale_exe, self._range_file, self._train_file,
                                                  self._scaled_file)
        Popen(cmd, shell=True, stdout=PIPE).communicate()

    def fit(self, X, y):  # todo: normalize
        class_weight = self.class_weight
        gamma = self.gamma
        if self.class_weight is not None:
            class_weight = self._compute_class_weight()
        if self.gamma is None:
            gamma = 1 / len(set(y))

        options = self._set_svm_options(class_weight, gamma)
        y = self._encode_y(y)

        # scale
        file = self._train_file
        self._convert_to_input_file(file, X, y)
        self._scale_input_file()

        # train
        file = self._scaled_file
        cmd = '{} {} "{}" "{}"'.format(self._svmtrain_exe, options, file, self._model_file)
        Popen(cmd, shell=True, stdout=PIPE).communicate()

    def predict(self, X):  # todo: continue!
        options = '-b {}'.format(self.probability)
        if self.quiet_mode:
            options += ' -q'

        file = self._train_file  # scale file?
        self._convert_to_input_file(file, X)
        self._scale_input_file()

        file = self._scaled_file
        cmd = f'{self._svmpredict_exe} {options} "{file}" "{self._model_file}" "{self._predict_file}"'
        Popen(cmd, shell=True).communicate()
        with open(self._predict_file, 'r') as f:
            if self.probability == 0:
                pred_labels = [int(s) for s in f.readlines()]
                pred_values = []
            else:
                x = np.array([s.split() for s in f.readlines()][1:]).astype(np.float)  # leaving the firs line
                pred_labels = x[:, 0].astype(np.int)
                pred_values = x[:, 1:]
        pred_labels = self._decode_y(pred_labels)
        return pred_labels


if __name__ == '__main__':
    _one_hot_encode_example()
    pass

import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import ai
from config import *
from preprocess import OfflineDataPreprocessor, SubjectKFold


class BCISystem(object):

    def __init__(self, base_dir="", window_length=0.5, window_step=0.25):
        """ Constructor for BCI system

        Parameters
        ----------
        base_dir: str
            absolute path to base dir of database
        window_length: float
            length of eeg processor window in seconds
        window_step: float
            window shift in seconds
        """
        self._base_dir = base_dir
        self._window_length = window_length
        self._window_step = window_step
        self._proc = None

    def offline_processing(self, db_name='physionet', feature='avg_column', epoch_tmin=0, epoch_tmax=3, use_drop_subject_list=True,
                           fast_load=True, subj_n_fold_num=None):
        self._proc = OfflineDataPreprocessor(self._base_dir, epoch_tmin, epoch_tmax, use_drop_subject_list,
                                             self._window_length, self._window_step, fast_load)
        if db_name == 'physionet':
            self._proc.use_physionet()
            labels = [REST, LEFT_HAND, RIGHT_HAND, BOTH_LEGS, BOTH_HANDS]
        else:
            raise NotImplementedError('Processor for {} is not implemented'.format(db_name))

        self._proc.run(feature=feature)
        kfold = SubjectKFold(subj_n_fold_num)

        for train_x, train_y, test_x, test_y, subject in kfold.split(self._proc):
            t = time.time()
            # svm = ai.SVM(C=.7, cache_size=2000, random_state=12)
            svm = ai.LinearSVM(C=1, random_state=12, max_iter=10000)
            svm.set_labels(labels)
            svm.fit(train_x, train_y, 'shrink rest')
            t = time.time() - t
            print("Training elapsed {} seconds.".format(t))

            y_pred = svm.predict(test_x)

            class_report = classification_report(test_y, y_pred)
            conf_martix = confusion_matrix(test_y, y_pred)
            acc = accuracy_score(test_y, y_pred)

            print("Classification report for subject{}:".format(subject))
            print("classifier %s:\n%s\n"
                  % (self, class_report))
            print("Confusion matrix:\n%s\n" % conf_martix)
            print("Accuracy score: {}\n".format(acc))

    def online_processing(self):
        raise NotImplementedError("Online processing is not implemented...")


if __name__ == '__main__':
    base_dir = "D:/BCI_stuff/databases/"  # MTA TTK
    # base_dir = 'D:/Csabi/'  # Home
    # base_dir = "D:/Users/Csabi/data/"  # ITK
    # base_dir = "/home/csabi/databases/"  # linux

    bci = BCISystem(base_dir)
    bci.offline_processing(feature='avg_column')

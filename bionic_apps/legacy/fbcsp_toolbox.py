# source: https://fbcsptoolbox.github.io/
import numpy as np
import scipy.linalg
import scipy.signal as signal
from sklearn.base import BaseEstimator, TransformerMixin


class CSP(TransformerMixin, BaseEstimator):
    def __init__(self, m_filters=4):
        self.m_filters = m_filters
        self.eig_vectors = None

    def fit(self, x_train, y_train):
        x_data = np.copy(x_train)
        y_labels = np.copy(y_train)
        n_trials, n_channels, n_samples = x_data.shape
        cov_x = np.zeros((2, n_channels, n_channels), dtype=np.float)
        for i in range(n_trials):
            x_trial = x_data[i, :, :]
            y_trial = y_labels[i]
            cov_x_trial = np.matmul(x_trial, np.transpose(x_trial))
            cov_x_trial /= np.trace(cov_x_trial)
            cov_x[y_trial, :, :] += cov_x_trial

        cov_x = np.asarray([cov_x[cls] / np.sum(y_labels == cls) for cls in range(2)])
        cov_combined = cov_x[0] + cov_x[1]
        eig_values, u_mat = scipy.linalg.eig(cov_combined, cov_x[0])
        sort_indices = np.argsort(abs(eig_values))[::-1]
        eig_values = eig_values[sort_indices]
        u_mat = u_mat[:, sort_indices]
        self.eig_vectors = np.transpose(u_mat)

        return self

    def transform(self, x_trial):
        if self.eig_vectors is None:
            raise RuntimeError('No filters available. Please first fit CSP '
                               'decomposition.')
        z_trial = np.matmul(self.eig_vectors, x_trial)
        z_trial_selected = z_trial[:self.m_filters, :]
        z_trial_selected = np.append(z_trial_selected, z_trial[-self.m_filters:, :], axis=0)
        sum_z2 = np.sum(z_trial_selected ** 2, axis=1)
        sum_z = np.sum(z_trial_selected, axis=1)
        var_z = (sum_z2 - (sum_z ** 2) / z_trial_selected.shape[1]) / (z_trial_selected.shape[1] - 1)
        sum_var_z = sum(var_z)
        return np.log(var_z / sum_var_z)


class FBCSP(TransformerMixin, BaseEstimator):
    def __init__(self, m_filters=4):
        self.m_filters = m_filters
        self.fbcsp_filters_multi = []
        self.csps = []

    def fit(self, x_train_fb, y_train):
        y_classes_unique = np.unique(y_train)
        n_classes = len(y_classes_unique)

        def get_csp(x_train_fb, y_train_cls):
            fbcsp_filters = {}
            for j in range(x_train_fb.shape[0]):
                x_train = x_train_fb[j, :, :, :]
                fbcsp_filters[j] = CSP(self.m_filters).fit(x_train, y_train_cls)
            return fbcsp_filters

        for i in range(n_classes):
            cls_of_interest = y_classes_unique[i]
            select_class_labels = lambda cls, y_labels: [0 if y == cls else 1 for y in y_labels]
            y_train_cls = np.asarray(select_class_labels(cls_of_interest, y_train))
            fbcsp_filters = get_csp(x_train_fb, y_train_cls)
            self.fbcsp_filters_multi.append(fbcsp_filters)

        return self

    def transform(self, x_data, class_idx=0):
        n_fbanks, n_trials, n_channels, n_samples = x_data.shape
        x_features = np.zeros((n_trials, self.m_filters * 2 * len(x_data)), dtype=np.float)
        for i in range(n_fbanks):
            for k in range(n_trials):
                x_trial = np.copy(x_data[i, k, :, :])
                csp_feat = self.fbcsp_filters_multi[i][k].transform(x_trial)
                for j in range(self.m_filters):
                    x_features[k, i * self.m_filters * 2 + (j + 1) * 2 - 2] = csp_feat[j]
                    x_features[k, i * self.m_filters * 2 + (j + 1) * 2 - 1] = csp_feat[-j - 1]

        return x_features


class FilterBank:
    def __init__(self, fs):
        self.fs = fs
        self.f_trans = 2
        self.f_pass = np.arange(4, 40, 4)
        self.f_width = 4
        self.gpass = 3
        self.gstop = 30
        self.filter_coeff = {}

    def get_filter_coeff(self):
        Nyquist_freq = self.fs / 2

        for i, f_low_pass in enumerate(self.f_pass):
            f_pass = np.asarray([f_low_pass, f_low_pass + self.f_width])
            f_stop = np.asarray([f_pass[0] - self.f_trans, f_pass[1] + self.f_trans])
            wp = f_pass / Nyquist_freq
            ws = f_stop / Nyquist_freq
            order, wn = signal.cheb2ord(wp, ws, self.gpass, self.gstop)
            b, a = signal.cheby2(order, self.gstop, ws, btype='bandpass')
            self.filter_coeff.update({i: {'b': b, 'a': a}})

        return self.filter_coeff

    def filter_data(self, eeg_data, window_details={}):
        n_trials, n_channels, n_samples = eeg_data.shape
        if window_details:
            n_samples = int(self.fs * (window_details.get('tmax') - window_details.get('tmin'))) + 1
        filtered_data = np.zeros((len(self.filter_coeff), n_trials, n_channels, n_samples))
        for i, fb in self.filter_coeff.items():
            b = fb.get('b')
            a = fb.get('a')
            eeg_data_filtered = np.asarray([signal.lfilter(b, a, eeg_data[j, :, :]) for j in range(n_trials)])
            if window_details:
                eeg_data_filtered = eeg_data_filtered[:, :, int((4.5 + window_details.get('tmin')) * self.fs):int(
                    (4.5 + window_details.get('tmax')) * self.fs) + 1]
            filtered_data[i, :, :, :] = eeg_data_filtered

        return filtered_data

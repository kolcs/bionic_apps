import mne
import numpy as np
from mne.utils import logger
from scipy import signal
from scipy.stats import kurtosis, zscore


def hurst(x):
    """Estimate Hurst exponent on a timeseries.

    The estimation is based on the second order discrete derivative.

    Parameters
    ----------
    x : 1D numpy array
        The timeseries to estimate the Hurst exponent for.

    Returns
    -------
    float
        The estimation of the Hurst exponent for the given timeseries.
    """
    y = np.cumsum(np.diff(x, axis=1), axis=1)

    b1 = [1, -2, 1]
    b2 = [1, 0, -2, 0, 1]

    # second order derivative
    y1 = signal.lfilter(b1, 1, y, axis=1)
    y1 = y1[:, len(b1) - 1:-1]  # first values contain filter artifacts

    # wider second order derivative
    y2 = signal.lfilter(b2, 1, y, axis=1)
    y2 = y2[:, len(b2) - 1:-1]  # first values contain filter artifacts

    s1 = np.mean(y1 ** 2, axis=1)
    s2 = np.mean(y2 ** 2, axis=1)

    return 0.5 * np.log2(s2 / s1)


def _freqs_power(data, sfreq, freqs):
    """A feature to evaluate channels/components"""

    fs, ps = signal.welch(data, sfreq,
                          nperseg=2 ** int(np.log2(10 * sfreq) + 1),
                          noverlap=0,
                          axis=-1)
    return np.sum([ps[..., np.searchsorted(fs, f)] for f in freqs], axis=0)


def _power_gradient(ica, source_data):
    """A feature to evaluate channels/components"""

    # Compute power spectrum
    f, Ps = signal.welch(source_data, ica.info['sfreq'])

    # Limit power spectrum to upper frequencies
    Ps = Ps[:, np.searchsorted(f, 25):np.searchsorted(f, 45)]

    # Compute mean gradients
    return np.mean(np.diff(Ps), axis=1)


def faster_bad_components(ica, epochs, thres=3, use_metrics=None, verbose=True):
    """Implements the third step of the FASTER algorithm.

    This function attempts to automatically mark bad ICA components by
    performing outlier detection.

    Parameters
    ----------
    ica : Instance of ICA
        The ICA operator, already fitted to the supplied Epochs object.
    epochs : Instance of Epochs
        The untransformed epochs to analyze.
    thres : float
        The threshold value, in standard deviations, to apply. A component
        crossing this threshold value is marked as bad. Defaults to 3.
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
            'eog_correlation', 'kurtosis', 'power_gradient', 'hurst',
            'median_gradient'
        Defaults to all of them.
    verbose: bool

    Returns
    -------
    bads : list of int
        The indices of the bad components.

    See also
    --------
    ICA.find_bads_ecg
    ICA.find_bads_eog
    """
    source_data = ica.get_sources(epochs).get_data().transpose(1, 0, 2)
    source_data = source_data.reshape(source_data.shape[0], -1)

    metrics = {
        'kurtosis': lambda x: kurtosis(
            np.dot(
                x.mixing_matrix_.T,
                x.pca_components_[:x.n_components_]),
            axis=1),
        'power_gradient': lambda x: _power_gradient(x, source_data),
        'hurst': lambda x: hurst(source_data),
        'median_gradient': lambda x: np.median(np.abs(np.diff(source_data)),
                                               axis=1),
        'line_noise': lambda x: _freqs_power(source_data,
                                             epochs.info['sfreq'], [50, 60]),
    }

    if use_metrics is None:
        use_metrics = metrics.keys()

    bads = []

    ica_scores = {}

    for m in use_metrics:
        scores = np.atleast_2d(metrics[m](ica))
        ica_scores[m] = [np.mean(scores[0]), np.std(scores[0])]
        for s in scores:
            b = find_outliers(s, thres)
            if verbose and len(b) > 0:
                logger.info('Bad by %s:\n\t%s' % (m, b))
            bads.append(b)

    return np.unique(np.concatenate(bads)).tolist(), ica_scores


def online_faster_bad_components(ica, epochs, ica_scores, thres=3, use_metrics=None, verbose=True):
    """Implements the third step of the FASTER algorithm for ONLINE use.

    Parameters
    ----------
    ica : Instance of ICA
        The ICA operator, already fitted to the supplied Epochs object.
    epochs : Instance of Epochs
        The untransformed epochs to analyze.
    ica_scores :
        the scores to ICA channels, computed during the offline filtering
    thres : float
        The threshold value, in standard deviations, to apply. A component
        crossing this threshold value is marked as bad. Defaults to 3.
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
            'eog_correlation', 'kurtosis', 'power_gradient', 'hurst',
            'median_gradient'
        Defaults to all of them.
    verbose: bool

    Returns
    -------
    bads : list of int
        The indices of the bad components.
    """
    source_data = ica.get_sources(epochs).get_data().transpose(1, 0, 2)
    source_data = source_data.reshape(source_data.shape[0], -1)

    metrics = {
        'kurtosis': lambda x: kurtosis(
            np.dot(
                x.mixing_matrix_.T,
                x.pca_components_[:x.n_components_]),
            axis=1),
        'power_gradient': lambda x: _power_gradient(x, source_data),
        'hurst': lambda x: hurst(source_data),
        'median_gradient': lambda x: np.median(np.abs(np.diff(source_data)),
                                               axis=1),
        'line_noise': lambda x: _freqs_power(source_data,
                                             epochs.info['sfreq'], [50, 60]),
    }

    if use_metrics is None:
        use_metrics = metrics.keys()

    bads = []

    for m in use_metrics:
        scores = np.atleast_2d(metrics[m](ica))
        ica_scores[m] = [np.mean(scores[0]), np.std(scores[0])]

        mean = ica_scores[m][0]
        dev = ica_scores[m][1]

        for s in scores:
            b = []
            for i, score in enumerate(s):
                if abs((score - mean) / dev) > thres:
                    b.append(i)

            if verbose and len(b) > 0:
                logger.info('Bad by %s:\n\t%s' % (m, b))
            bads.append(b)

    return np.unique(np.concatenate(bads)).tolist()


def find_outliers(X, threshold=3.0, max_iter=2):
    my_mask = np.zeros(len(X), dtype=np.bool)
    for _ in range(max_iter):
        X = np.ma.masked_array(X, my_mask)
        this_z = np.abs(zscore(X))
        local_bad = this_z > threshold
        my_mask = np.max([my_mask, local_bad], 0)
        if not np.any(local_bad):
            break

    bad_idx = np.where(my_mask)[0]
    return bad_idx


def faster_bad_channels(epochs, picks=None, thres=3, use_metrics=None, verbose=True):
    """Implements the first step of the FASTER algorithm.

    This function attempts to automatically mark bad EEG channels by performing
    outlier detection. It operated on epoched data, to make sure only relevant
    data is analyzed.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs for which bad channels need to be marked
    picks : list of int | None
        Channels to operate on. Defaults to EEG channels.
    thres : float
        The threshold value, in standard deviations, to apply. A channel
        crossing this threshold value is marked as bad. Defaults to 3.
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
        'variance', 'correlation', 'hurst', 'kurtosis', 'line_noise'
        Defaults to all of them.
    verbose: bool

    Returns
    -------
    bads : list of str
        The names of the bad EEG channels.
    """
    metrics = {
        'variance': lambda x: np.var(x, axis=1),
        'correlation': lambda x: np.nanmean(
            np.ma.masked_array(
                np.corrcoef(x),
                np.identity(len(x), dtype=bool)
            ),
            axis=0),
        'hurst': lambda x: hurst(x)
    }

    if picks is None:
        picks = mne.pick_types(epochs.info, meg=False, eeg=True, exclude=[])
    if use_metrics is None:
        use_metrics = metrics.keys()

    # Concatenate epochs in time
    data = epochs.get_data()
    data = data.transpose(1, 0, 2).reshape(data.shape[1], -1)
    data = data[picks]

    # Find bad channels
    bads = []
    parameters = {}

    for m in use_metrics:
        s = metrics[m](data)
        b = [epochs.ch_names[picks[i]] for i in find_outliers(s, thres)]
        if verbose and len(b) > 0:
            logger.info('Bad by %s:\n\t%s' % (m, b))
        bads.append(b)
        parameters[m] = s

    return np.unique(np.concatenate(bads)).tolist()


def _deviation(data):
    """Computes the deviation from mean for each channel in a set of epochs.

    This is not implemented as a lambda function, because the channel means
    should be cached during the computation.

    Parameters
    ----------
    data : ndarray
        The epochs (epochs x channels x samples).

    Returns
    -------
    dev : ndarray
        For each epoch, the mean deviation of the channels.
    """
    ch_mean = np.mean(data, axis=2)
    return ch_mean - np.mean(ch_mean, axis=0)


def faster_bad_epochs(epochs, picks=None, thres=3, use_metrics=None, verbose=True):
    """Implements the second step of the FASTER algorithm.

    This function attempts to automatically mark bad epochs by performing
    outlier detection.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to analyze.
    picks : list of int | None
        Channels to operate on. Defaults to EEG channels.
    thres : float
        The threshold value, in standard deviations, to apply. An epoch
        crossing this threshold value is marked as bad. Defaults to 3.
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
            'amplitude', 'variance', 'deviation'
        Defaults to all of them.
    verbose: bool

    Returns
    -------
    bads : list of int
        The indices of the bad epochs.
    """

    metrics = {
        'amplitude': lambda x: np.mean(np.ptp(x, axis=2), axis=1),
        'deviation': lambda x: np.mean(_deviation(x), axis=1),
        'variance': lambda x: np.mean(np.var(x, axis=2), axis=1),
    }

    if picks is None:
        picks = mne.pick_types(epochs.info, meg=False, eeg=True,
                               exclude='bads')
    if use_metrics is None:
        use_metrics = metrics.keys()

    data = epochs.get_data()[:, picks, :]

    bads = []
    for m in use_metrics:
        s = metrics[m](data)
        b = find_outliers(s, thres)
        if verbose and len(b) > 0:
            logger.info('Bad by %s:\n\t%s' % (m, b))
        bads.append(b)

    return np.unique(np.concatenate(bads)).tolist()


def faster_bad_channels_in_epochs(epochs, picks=None, thres=3, use_metrics=None, verbose=True):
    """Implements the fourth step of the FASTER algorithm.

    This function attempts to automatically mark bad channels in each epochs by
    performing outlier detection.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to analyze.
    picks : list of int | None
        Channels to operate on. Defaults to EEG channels.
    thres : float
        The threshold value, in standard deviations, to apply. An epoch
        crossing this threshold value is marked as bad. Defaults to 3.
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
            'amplitude', 'variance', 'deviation', 'median_gradient'
        Defaults to all of them.
    verbose: bool

    Returns
    -------
    bads : list of lists of int
        For each epoch, the indices of the bad channels.
    """

    metrics = {
        'amplitude': lambda x: np.ptp(x, axis=2),
        'deviation': lambda x: _deviation(x),
        'variance': lambda x: np.var(x, axis=2),
        'median_gradient': lambda x: np.median(np.abs(np.diff(x)), axis=2)
    }

    if picks is None:
        picks = mne.pick_types(epochs.info, meg=False, eeg=True,
                               exclude='bads')
    if use_metrics is None:
        use_metrics = metrics.keys()

    data = epochs.get_data()[:, picks, :]

    bads = [[] for i in range(len(epochs))]
    for m in use_metrics:
        s_epochs = metrics[m](data)
        for i, s in enumerate(s_epochs):
            b = [epochs.ch_names[picks[j]] for j in find_outliers(s, thres)]
            if verbose and len(b) > 0:
                logger.info('Epoch %d, Bad by %s:\n\t%s' % (i, m, b))
            bads[i].append(b)

    for i, b in enumerate(bads):
        if len(b) > 0:
            bads[i] = np.unique(np.concatenate(b)).tolist()

    return bads


def run_faster(epochs, thresholds=None, copy=True, apply_frequency_filter=True,
               filter_low=0.5, filter_high=45, verbose=True, apply_avg_reference=True):
    """
    Applies all step of the FASTER artefact rejection method on the input data, with the given parameters

    Parameters
    ----------
    epochs : mne.Epoch
        The epochs to analyze
    thresholds : float or int or array
        The threshold value, in standard deviations, to apply. An epoch
        crossing this threshold value is marked as bad. Defaults to 3.
        Can be given as an array of 4 elements, where the n-th element is the threshold of the n-th
        step of FASTER algorithm. If a threshold is set to 0, it means that actual step will be left out
    copy : bool
        Determines if to work on the actual data, or to make a copy of it, and return with it
    apply_frequency_filter : bool
        Determines weather an initial frequency filter should be applied before the FASTER algorithm
    apply_avg_reference : bool
        If true, as the last step, the average of all electrodes applied as the reference
    filter_low : float
        Lower bound of frequency filter
    filter_high : float
        Upper bound of frequency filter
    verbose : bool
        If true, display more information on the command line
    """
    if thresholds is None:
        thresholds = [3, 3, 3, 3]
    elif isinstance(thresholds, (int, float)):
        thresholds = [thresholds] * 4

    assert len(thresholds) == 4, f'There are 4 steps in FASTER algorithm. ' \
                                 f'{len(thresholds)} thresholds were defined instead.'

    epochs.load_data()

    if copy:
        epochs = epochs.copy()

    if apply_frequency_filter:
        epochs.filter(filter_low, filter_high)

    droped_epochs = -1

    # Step one - save bad channels to lsl, interpolate here and even lsl

    if thresholds[0] > 0:
        if verbose:
            logger.info('Step 1: mark bad channels')
        epochs.info['bads'] += faster_bad_channels(epochs, thres=thresholds[0], verbose=verbose)
        bads = list.copy(epochs.info['bads'])
        # epochs.interpolate_bads(reset_bads=False, verbose=False, mode='fast')
    else:
        if verbose:
            logger.info('Step 1 - mark bad channels - is dismissed')
        bads = list.copy(epochs.info['bads'])

    # Step two - only offline, drop bad epochs
    if thresholds[1] > 0:
        if verbose:
            logger.info('Step 2: mark bad epochs')
        bad_epochs = faster_bad_epochs(epochs, thres=thresholds[1], verbose=verbose)
        good_epochs = list(set(range(len(epochs))).difference(set(bad_epochs)))
        epochs = epochs[good_epochs]
        droped_epochs = len(bad_epochs) / len(good_epochs)
    else:
        if verbose:
            logger.info('Step 2 - mark bad epochs - is dismissed')

    # Step three - save ica weights, component parameters mean and st. deviation to calculate Z-score lsl

    if thresholds[2] > 0:
        if verbose:
            logger.info('Step 3: mark bad ICA components')
        ica = mne.preprocessing.ICA(max_iter="auto")
        ica.fit(epochs)

        eog_inds, _ = ica.find_bads_eog(epochs, ch_name='Fp1', threshold=thresholds[2], verbose=verbose)
        bad_components, ica_scores = faster_bad_components(ica, epochs, verbose=verbose, thres=thresholds[2])
        bad_components = np.unique(np.append(bad_components, eog_inds))
        epochs = ica.apply(epochs, exclude=bad_components)

    else:
        if verbose:
            logger.info('Step 3 - ICA - is dismissed')
        ica, ica_scores = None, None

    # Step four
    if thresholds[3] > 0:
        if verbose:
            logger.info('Step 4: mark bad channels for each epoch')
        bad_channels_per_epoch = faster_bad_channels_in_epochs(epochs, thres=thresholds[3], verbose=verbose)
        for i, b in enumerate(bad_channels_per_epoch):
            if len(b) > 0:
                epoch = epochs[i]
                epoch.info['bads'] += b
                epoch.interpolate_bads(verbose=verbose)
                epochs._data[i, :, :] = epoch._data[0, :, :]

    # Now that the data is clean, apply average reference
    if apply_avg_reference:
        # epochs.info['custom_ref_applied'] = False
        epochs, _ = mne.io.set_eeg_reference(epochs, verbose=verbose)
        epochs.apply_proj()

    if droped_epochs != -1:
        logger.info(f'\nAmount of dropped epochs {droped_epochs * 100:.2f}%')

    return epochs, bads, ica, ica_scores


def online_faster(data, bad_channels, ica, ica_scores, apply_frequency_filter=True, filter_low=0.5, filter_high=45,
                  thresholds=None, verbose=False, apply_avg_reference=True):
    """Online FASTER algorithm

    Parameters
    ----------
    data : mne.Epochs, mne.EpochsArray
        An mne epoch to filter
    bad_channels :
        The offline marked globally bad channels
    ica :
        ICA weights, computed during offline filtering
    ica_scores :
        the scores to ICA channels, computed during the offline filtering
    apply_frequency_filter : bool
        Determines weather an initial frequency filter should be applied
        before the FASTER algorithm
    filter_low : float
        Lower bound of frequency filter
    filter_high : float
        Upper bound of frequency filter
    verbose : bool
        If true, display more information on the command line
    thresholds : float or int or array
        The threshold value, in standard deviations, to apply. An epoch
        crossing this threshold value is marked as bad. Defaults to 3.
        Can be given as an array of 2 elements, where the n-th element is
        the threshold of the n-th step of FASTER algorithm. If a threshold
        is set to 0, it means that actual step will be left out
    apply_avg_reference : bool
        If true, as the last step, the average of all electrodes applied
        as the reference.

    Returns
    -------
    data : mne.Epochs
        The filtered epoch
    """

    if thresholds is None:
        thresholds = [3, 3]
    elif isinstance(thresholds, (int, float)):
        thresholds = [thresholds] * 2

    if apply_frequency_filter:
        data.filter(filter_low, filter_high, verbose=verbose)

    if thresholds[0] > 0:
        data.info['bads'] = bad_channels
        data.info['bads'] += faster_bad_channels(data, thres=thresholds[0], verbose=verbose)
        data.interpolate_bads(reset_bads=True, verbose=verbose, mode='fast')

    if thresholds[1] > 0:
        eog_inds, _ = ica.find_bads_eog(data, ch_name='Fp1', threshold=thresholds[1], verbose=verbose)
        bad_components = online_faster_bad_components(ica, data, ica_scores, verbose=verbose)
        bad_components = np.unique(np.append(bad_components, eog_inds))
        ica.apply(data, exclude=bad_components)

    # Data is clean, apply average reference
    if apply_avg_reference:
        # data.info['custom_ref_applied'] = False
        epochs, _ = mne.io.set_eeg_reference(data, verbose=verbose)
        epochs.apply_proj()

    return data


class ArtefactFilter:

    def __init__(self, apply_frequency_filter=True, filter_low=0.5, filter_high=45,
                 thresholds=None, apply_avg_reference=True, verbose=True):
        """FASTER algorithm for artefact filtering

        The object, responsible for storing parameters (such as ICA weights)
        between offline and lsl FASTER filtering

        Parameters
        ----------
        apply_frequency_filter : bool
            Determines weather an initial frequency filter should be applied
            before the FASTER algorithm
        filter_low : float
            Lower bound of frequency filter
        filter_high : float
            Upper bound of frequency filter
        thresholds : float or int or List[float] or List[int]
            The threshold value, in standard deviations, to apply. An epoch
            crossing this threshold value is marked as bad. Defaults to 3.
            Can be given as an array of 4 elements, where the n-th element
            is the threshold of the n-th step of FASTER algorithm. If a
            threshold is set to 0, it means that actual step will be left out.
        verbose :
            If true, display more information on the command line during filtering
        apply_avg_reference : bool
            If true, as the last step, the average of all electrodes applied
            as the reference
        """

        self._info = None
        self._ica = None
        self._ica_scores = None
        self._params = None
        self.bad_channels = None
        self._apply_frequency_filter = apply_frequency_filter
        self._filter_low = filter_low
        self._filter_high = filter_high
        if thresholds is not None:
            if isinstance(thresholds, (int, float)):
                self.thresholds = [thresholds] * 4
            else:
                self.thresholds = thresholds
        else:
            self.thresholds = [3] * 4

        self._verbose = verbose
        self._apply_avg_reference = apply_avg_reference

    def offline_filter(self, epochs):
        """Offline Faster algorithm

        Filters the input epochs, and saves the parameters (such as ICA weights),
        for the possibility of lsl filtering

        Parameters
        ----------
        epochs : mne.Epochs
            The epochs to analyze

        Returns
        -------
        mne.Epochs
            The filtered epoch
        """

        epochs, self.bad_channels, self._ica, self._ica_scores \
            = run_faster(epochs, apply_frequency_filter=self._apply_frequency_filter,
                         filter_low=self._filter_low, filter_high=self._filter_high,
                         verbose=self._verbose, thresholds=self.thresholds,
                         apply_avg_reference=self._apply_avg_reference
                         )
        self._info = epochs.info
        return epochs

    def online_filter(self, data):
        """Online Faster

        Filters the input data with the FASTER algorithm, with the saved parameters of
        during offline filtering.

        Parameters
        ----------
        data : ndarray, mne.Epochs
            A data to filter, (with the previously saved parameters during offline filtering)
        """

        if self._ica is None or self._info is None or self.bad_channels is None:
            raise Exception("offline filter should be applied before")

        if len(data.shape) == 2:
            data = np.expand_dims(data, 0)
        epoch = mne.EpochsArray(data, self._info)

        filtered_epoch = online_faster(epoch, self.bad_channels, self._ica, self._ica_scores,
                                       apply_frequency_filter=self._apply_frequency_filter,
                                       filter_low=self._filter_low,
                                       filter_high=self._filter_high,
                                       thresholds=[self.thresholds[0], self.thresholds[2]],
                                       apply_avg_reference=self._apply_avg_reference,
                                       verbose=self._verbose)
        return filtered_epoch.get_data()

    def mimic_online_filter(self, epochs):
        epochs = epochs.copy()
        epochs.load_data()
        for i in range(len(epochs)):
            data = epochs[i].get_data()
            epochs._data[i, :, :] = self.online_filter(data)[0, :, :]
        return epochs

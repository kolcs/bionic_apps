"""
Online BCI

@license: PPKE ITK, TTK
@author: Köllőd Csaba, kollod.csaba@ikt.ppke.hu
"""

import numpy as np
from pylsl import StreamInlet, resolve_stream
from scipy import signal


class SignalReceiver:

    def __init__(self):
        self._inlet = None
        self.fs = None  # sampling frequency rate
        self.electrodes = list()
        # self.n_channels = 0
        self._init_inlet()

    def _init_inlet(self):
        print("looking for an EEG stream...")
        streams = resolve_stream('type', 'EEG')
        inlet = StreamInlet(streams[0])
        print("EEG stream found!")
        self._inlet = inlet  # Keep it None until inlet is ready
        self._load_init_info()

    def _load_init_info(self):
        self.fs = self._load_sampling_frequency()
        self.electrodes = self._load_electrodes()
        # self.n_channels = self._inlet.info.channel_count()

    def _load_sampling_frequency(self):
        return self._inlet.info().nominal_srate()  # Hz

    def _load_electrodes(self):
        electrodes = []
        info = self._inlet.info()
        ch = info.desc().child("channels").child("channel")
        for _ in range(info.channel_count()):
            electrodes.append(ch.child_value("label"))
            ch = ch.next_sibling()
        return electrodes

    def get_sample(self, timeout=32000000.0):
        return self._inlet.pull_sample(timeout=timeout)

    def get_chunk(self):
        return self._inlet.pull_chunk()


class DSP(SignalReceiver):

    def __init__(self, use_filter=False, order=5, l_freq=1):
        super(DSP, self).__init__()
        self._eeg = list()
        self._filt_eeg = list()  # change it to deque + copy()?
        self._timestamp = list()
        self._filter_signal = use_filter

        if use_filter:
            self._sos = signal.butter(order, l_freq, btype='high', output='sos', fs=self.fs)
            self._zi = np.array([signal.sosfilt_zi(self._sos) for _ in range(len(self.electrodes))])
            self._zi = np.transpose(self._zi, (1, 2, 0))

    def get_eeg_window_in_chunk(self, window_length=1.0):
        eeg_samples, timestamps = self.get_chunk()

        if len(timestamps) == 0:
            return None, None

        if self._filter_signal:
            eeg_samples, self._zi = signal.sosfilt(self._sos, eeg_samples, axis=0, zi=self._zi)

        self._eeg.extend(eeg_samples)
        self._timestamp.extend(timestamps)
        win = int(self.fs * window_length)
        timestamp = self._timestamp[-win:]
        eeg = self._eeg[-win:]
        if len(timestamp) < win:
            return None, None
        return timestamp, np.transpose(eeg)

    def get_recorded_signal(self):
        return np.transpose(self._eeg)

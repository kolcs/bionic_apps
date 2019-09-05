"""
Online BCI

@license: PPKE ITK, MTA TTK
@author: Köllőd Csaba, kollod.csaba@ikt.ppke.hu
"""

from pylsl import StreamInlet, resolve_stream
import numpy as np
import threading
from scipy.signal import butter, lfilter, lfilter_zi
from online.plotter import EEGPlotter


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


class DSP(SignalReceiver):

    def __init__(self):
        super(DSP, self).__init__()
        self._eeg = list()
        self._filt_eeg = list()
        self._timestamp = list()
        self._stop_recording = False
        self._lock = threading.Lock()
        self._ch_list = None
        self._plotter = None

    def get_eeg_window(self, wlength=1.0, return_label=False):
        win = int(self.fs * wlength)
        self._lock.acquire()
        eeg = self._eeg[-win:]
        self._lock.release()
        if return_label:
            label = eeg[0][-1] if len(eeg) > 0 else None
            data = np.transpose(eeg)[:-1, :] if len(eeg) > 0 else []
            return data, label
        return np.transpose(eeg)

    def start_signal_recording(self):
        thread = threading.Thread(target=self._record_signal, daemon=True)
        thread.start()

    def _record_signal(self):
        # self._save_init_data()
        self._stop_recording = False
        self._reset_data()

        while not self._stop_recording:
            EEG_sample, timestamp = self.get_sample()

            self._lock.acquire()
            self._eeg.append(EEG_sample)
            self._timestamp.append(timestamp)  # + self._inlet.time_correction())

            self._lock.release()

    def process_singal(self, wlength=1):
        self._stop_recording = False
        win = int(self.fs * wlength)

        # loading up buffer
        while len(self._eeg) < win:
            eeg_sample, timestamp = self.get_sample()
            self._eeg.append(eeg_sample)

        # todo: extend to more eeg bands
        # processing signal
        low = 8
        high = 12
        order = 5
        b, a = butter(order, (low, high), btype='bandpass', fs=self.fs)
        z_init = lfilter_zi(b, a)
        z = [z_init for _ in range(len(self._eeg[0]))]
        z = np.transpose(np.array(z))

        while not self._stop_recording:
            eeg_sample, timestamp = self.get_sample()
            self._lock.acquire()
            self._eeg.pop(0)
            self._eeg.append(eeg_sample)
            self._filt_eeg, z = lfilter(b, a, np.array(self._eeg), axis=0, zi=z)  # eeg: data x channel
            # self._filt_eeg = np.power(self._filt_eeg, 2)
            self._lock.release()

    def run_online_plotter(self, channels):
        self._select_channels_for_plot(channels)
        self._plotter = EEGPlotter(plot_size=(len(channels), 2))
        self._plotter.add_data_source(self)
        thread = threading.Thread(target=self.process_singal, daemon=True)
        thread.start()
        self._plotter.run()

    def _select_channels_for_plot(self, channels=('Cpz', 'Cp1', 'Cp2', 'Cp5', 'Cp6')):
        assert self.electrodes is not None, "missing electrode names"
        self._ch_list = [self.electrodes.index(ch) for ch in channels]

    def get_data(self):
        assert self._ch_list is not None, "Channels are not selected for online plot!"
        eeg = list()
        if len(self._filt_eeg) > 0:
            self._lock.acquire()
            eeg = np.transpose(np.array(self._eeg))[self._ch_list, :]
            filt_eeg = np.transpose(self._filt_eeg)[self._ch_list, :]
            self._lock.release()
            for i, x in enumerate(filt_eeg):
                eeg = np.insert(eeg, 2 * i + 1, x, axis=0)
        return eeg

    def stop_signal_recording(self):
        self._stop_recording = True

    def _reset_data(self):
        self._eeg = list()
        self._timestamp = list()


if __name__ == '__main__':
    channels = ['Cpz', 'Cp1', 'Cp2', 'Cp5', 'Cp6']
    # channels = ['Cpz']
    bci = DSP()
    bci.run_online_plotter(channels)
    # bci.process_singal() # do not use it with bci.run_olnine_plotter!

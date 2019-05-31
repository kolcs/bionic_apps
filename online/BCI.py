# -*- coding: utf-8 -*-
'''
Created on 13 Apr 2018

Online BCI

@license: PPKE ITK, MTA TTK
@author: Köllőd Csaba
'''

from pylsl import StreamInlet, resolve_stream
import numpy as np
import threading
import matplotlib.pyplot as plt
import time
from scipy.signal import butter, lfilter, lfilter_zi


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
        print("OK")
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

    def get_sample(self):
        return self._inlet.pull_sample()


class DSP(SignalReceiver):

    def __init__(self):
        super(DSP, self).__init__()
        self._eeg = list()
        self._filt_eeg = list()
        self._timestamp = list()
        self._stop_recording = False
        self._lock = threading.Lock()

    def get_eeg_window(self, wlength=1):
        win = int(self.fs * wlength)
        self._lock.acquire()
        eeg = self._eeg[-win:]
        self._lock.release()
        eeg = np.transpose(np.array(eeg))
        return eeg

    def start_signal_recording(self):
        thread = threading.Thread(target=self._record_signal)
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
            EEG_sample, timestamp = self.get_sample()
            self._eeg.append(EEG_sample)

        # processing signal
        low = 5
        high = 13
        order = 5
        b, a = butter(order, (low, high), btype='bandpass', fs=self.fs)
        z_init = lfilter_zi(b, a)
        z = [z_init for _ in range(len(self._eeg))]

        graphs = list()

        while not self._stop_recording:
            self._filt_eeg, z = lfilter(b, a, np.array(self._eeg), zi=z)  # eeg: data x channel
            self.live_plotter(graphs)
            self._eeg.pop(0)
            EEG_sample, timestamp = self.get_sample()
            self._eeg.append(EEG_sample)

    def live_plotter(self, graphs, pause_time=0.01):
        # data = insert numpy data: merge eeg with filtered
        eeg = np.transpose(np.array(self._eeg))
        filt_eeg = np.transpose(np.array(self._filt_eeg))
        for i, x in enumerate(filt_eeg):
            eeg = np.insert(eeg, 2 * i + 1, x, axis=0)
            # eeg.insert(i * 2 + 1, self._filt_eeg[i])

        data = eeg  # np.transpose(np.array(eeg))
        labels = self.electrodes

        # def thread_plot():  # graphs, pause_time, labels, data):
        if graphs == []:
            plt.ion()
            fig = plt.figure()
            for i in range(len(labels) * 2):
                ax = fig.add_subplot(len(labels), 2, i + 1)
                line, = ax.plot(data[i])
                graphs.append(line)
                plt.ylabel(labels[i // 2])
                plt.show()
        else:
            print("------")
            for i in range(len(labels) * 2):
                graphs[i].set_ydata(data[i])
            plt.pause(1/self.fs)  # pause_time)

        # thread = threading.Thread(target=thread_plot)  # , args=(graphs, pause_time, self.electrodes, data))
        # thread.start()

    # def f_anim(self, ev, n_channel, subplts):
    #     eeg = self._eeg
    #     filt_eeg = self._filt_eeg
    #     print(eeg, filt_eeg)
    #     if len(eeg) > 0:  # todo: get the real signals!!!!
    #         eeg = np.transpose(np.array(eeg))
    #         filt_eeg = np.transpose(np.array(filt_eeg))
    #         for i in range(n_channel):
    #             subplts[i * 2].clear()
    #             subplts[i * 2].plot(eeg[i, :])
    #             if np.size(filt_eeg, 0) > 0:
    #                 subplts[i * 2 + 1].clear()
    #                 subplts[i * 2 + 1].plot(filt_eeg[i, :])
    #
    # def use_animation(self, n_channel=24, signal='both'):
    #     import matplotlib.pyplot as plt
    #     import matplotlib.animation as animation
    #
    #     if signal == 'both':
    #         signal = ['raw', 'filtered']
    #     else:
    #         signal = [signal]
    #     fig = plt.figure()
    #
    #     subplts = list()
    #     index = 1
    #     for i in range(n_channel):
    #         for j in range(len(signal)):
    #             subplts.append(fig.add_subplot(n_channel, len(signal), index))
    #             index += 1
    #
    #     ani = animation.FuncAnimation(fig, self.f_anim, fargs=(n_channel, subplts),
    #                                   interval=self.fs)
    #     plt.show()

    def stop_signal_recording(self):
        self._stop_recording = True

    def _reset_data(self):
        self._eeg = list()
        self._timestamp = list()


if __name__ == '__main__':
    bci = DSP()
    # bci.use_animation()
    bci.process_singal()

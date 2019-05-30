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
import time


class SignalReceiver:
    _inlet = None
    _stop_recording = False
    _eeg = list()
    _timestamp = list()
    # _index = 0
    # _trigger = False
    # _dir = "train/"
    fs = None  # sampling frequency rate
    electrodes = list()
    # row_col = None

    def __init__(self):
        self._init_inlet()
        self._lock = threading.Lock()
        # self._analyser = SignalAnalyser()

    def _init_inlet(self):
        def init_inlet():
            print("looking for an EEG stream...")
            streams = resolve_stream('type', 'EEG')
            inlet = StreamInlet(streams[0])
            print("OK")
            self._inlet = inlet  # Keep it None until inlet is ready
            self._load_init_info()

        thread = threading.Thread(target=init_inlet, daemon=True)
        thread.start()

    def _load_init_info(self):
        self._fs = self._load_sampling_frequency()
        self._electrodes = self._load_electrodes()

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

    # def set_data_directory(self, directory):
    #     self._dir = directory
    #     self._reset_index()
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)

    def stream_connected(self):
        if self._inlet:
            return True
        else:
            return False

    def get_sample(self):
        return self._inlet.pull_sample()

    def get_eeg_window(self, wlength=1):
        win = int(self._fs * wlength)
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
            EEG_sample, timestamp = self._inlet.pull_sample()

            self._lock.acquire()
            self._eeg.append(EEG_sample)
            self._timestamp.append(timestamp)  # + self._inlet.time_correction())

            # if self._trigger:
            #     self._trigger = False
            #     self.marker.append(self.trigger_type)
            # else:
            #     self.marker.append([-1, -1])
            self._lock.release()

    def stop_signal_recording(self):
        self._stop_recording = True

    # def trigger_signal(self, trigger_type):
    #     self._lock.acquire()
    #     self._trigger = True
    #     if trigger_type[:-1] == "row":
    #         self.trigger_type = [int(trigger_type[-1]), -1]
    #     elif trigger_type[:-1] == "col":
    #         self.trigger_type = [-1, int(trigger_type[-1])]
    #     self._lock.release()

    def _reset_data(self):
        self._eeg = list()
        # self.marker = list()
        self._timestamp = list()

    # def _reset_index(self):
    #     self._index = 0

    # def save_session(self):
    #     thread = threading.Thread(target=self._save_sess_in_lock)
    #     thread.start()

    # def _convert_data_to_numpy(self):
    #     self._eeg = np.transpose(np.array(self._eeg))
    #     self.marker = np.transpose(np.array(self.marker))

    # def _save_sess_in_lock(self):
    #     self._lock.acquire()
    #     self._convert_data_to_numpy()
    #     self._save_sess()
    #     self._lock.release()

    # def _save_sess(self):
    #     with open(self._dir + 'eeg' + str(self._index) + F_EXT, 'wb') as f:
    #         pickle.dump(self.eeg, f)
    #     with open(self._dir + 'marker' + str(self._index) + F_EXT, 'wb') as f:
    #         pickle.dump(self.marker, f)
    #     with open(self._dir + 'timestamp' + str(self._index) + F_EXT, 'wb') as f:
    #         pickle.dump(self.timestamp, f)
    #     self._index += 1
    #     self._reset_data()

    # def _save_init_data(self):
    #     self.save_data(self._electrodes, 'electrodes')
    #     self.save_data(self._fs, 'FS')
    #
    # def save_char_dict(self, char_dict, filename):
    #     self.save_data(char_dict, filename)
    #     self._analyser.set_basics(self._fs, char_dict, self._electrodes)
    #
    # def save_data(self, data, filename):
    #     with open(self._dir + filename + F_EXT, 'wb') as f:
    #         pickle.dump(data, f)
    #
    # def set_training(self, train):
    #     self._analyser.training = train
    #
    # def print_stat(self):
    #     self._analyser.print_stat()
    #
    # def analyse_signal(self, char):
    #     def analyse_signal():
    #         self._lock.acquire()
    #         self._convert_data_to_numpy()
    #         eeg = self.eeg
    #         marker = self.marker
    #         self._save_sess()
    #         self._lock.release()
    #         self.row_col = self._analyser.start_online_analysis(eeg, marker, char)
    #
    #     thread = threading.Thread(target=analyse_signal)
    #     thread.start()


if __name__ == '__main__':
    bci = SignalReceiver()
    while not bci.stream_connected():
        print('...')
        time.sleep(0.1)
    bci.start_signal_recording()
    while True:
        print(np.shape(bci.get_eeg_window()))

import numpy as np
from scipy.signal import butter, lfilter, find_peaks, sosfilt

from ..databases import GameDB
from ..handlers.gui import select_files_in_explorer
from ..preprocess.io import get_epochs_from_files


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, fmode='ba'):
    if fmode == 'ba':
        b, a = butter(order, (lowcut, highcut), btype='bandpass', fs=fs)
        y = lfilter(b, a, data)
    elif fmode == 'sos':
        sos = butter(order, (lowcut, highcut), btype='bandpass', output=fmode, fs=fs)
        y = sosfilt(sos, data)
    else:
        raise AttributeError('Filter mode {} is not defined'.format(fmode))
    return y


def _test_blinking_detection(filename, blink_list=None):
    epoch_length = 4
    baseline = tuple([None, 0.1])
    task_dict = GameDB.TRIGGER_TASK_CONVERTER  # {'Rest': 1, 'left fist/both fists': 2, 'right fist/both feet': 3}
    epochs, fs = get_epochs_from_files(filename,
                                       task_dict=task_dict,
                                       epoch_tmin=0, epoch_tmax=epoch_length, baseline=baseline, get_fs=True,
                                       prefilter_signal=True)
    epochs.load_data()
    ch_list = ['Fp1', 'Fp2', 'Af7', 'Af8', 'Afz']
    epochs.pick_channels(ch_list)
    # epochs.plot(block=True)  # check blinks visually here

    detected = list()
    for i, ep in enumerate(epochs):
        if is_there_blinking(ep, fs, threshold=4, ch_list=ch_list):
            print('Epoch {} contains blinking'.format(i + 1))
            detected.append(i + 1)

    if blink_list is not None:
        print('\nSummary:')
        missed_blinks = [b for b in blink_list if b not in detected]
        wrongly_detected = [b for b in detected if b not in blink_list]
        print("Missed blinks: {}".format(missed_blinks))
        print("Detected but not blink: {}".format(wrongly_detected))
        epochs.plot(block=True)  # check the error...


def is_there_blinking(eeg, fs, threshold=4, ch_list=None):
    filt_data = butter_bandpass_filter(eeg, .5, 30, fs, order=5, fmode='ba')
    is_there_blink = False
    for ch_num, ch_data in enumerate(filt_data):
        ind, peaks = find_peaks(ch_data, height=0)
        avg_peak = np.mean(peaks['peak_heights'])
        ind_, peaks_ = find_peaks(ch_data, height=avg_peak * threshold)
        if len(ind_) != 0:
            if ch_list is not None:
                print("\tChannel {} contains blinking.".format(ch_list[ch_num]))
            is_there_blink = True

    return is_there_blink


if __name__ == '__main__':
    # on: Game/mixed/subject1
    blink_list = [2, 4, 7, 9, 10, 11, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30, 33, 35, 37, 39, 40, 43, 44,
                  47, 48, 49, 50, 51, 52, 54, 55, 57, 58, 61, 62, 63, 64, 68, 70, 72, 74, 75, 77, 79, 81, 82, 84, 86,
                  87,
                  88, 89, 90, 92, 94, 95, 97, 99, 100, 101, 103, 105, 106, 107, 109, 111, 113, 115, 117, 119, 120, 121,
                  122, 123, 125, 127, 128, 129, 131, 133, 134, 135, 137, 138, 139, 140, 141]
    filename = select_files_in_explorer()[0]
    _test_blinking_detection(filename, blink_list)

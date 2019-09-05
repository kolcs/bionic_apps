"""Example program to demonstrate how to send a multi-channel time-series
with proper meta-data to LSL."""

import time
from pylsl import StreamInfo, StreamOutlet, local_clock
import numpy as np

from preprocess.ioprocess import open_raw_file

M_BRAIN_TRAIN = 'mBrainTrain'


def get_wave(f, t):
    return np.sin(f * np.pi * 2 * t)


def get_artificial_data(ch_num, length, fs, max_dif=0.01):
    t = np.linspace(0, length, length * fs)
    signal = get_wave(5, t) + get_wave(11, t) + get_wave(17, t) + get_wave(27, t)
    signal = signal / np.linalg.norm(signal)
    data = [signal for _ in range(ch_num)]
    data = np.array(data) * max_dif
    return data


def get_data_with_labels(raw):  # only for pilot data
    from mne import events_from_annotations
    fs = raw.info['sfreq']
    events, event_id = events_from_annotations(raw)

    to = [i for i, el in enumerate(events[:, 2]) if el == 12][3:]  # end stim
    frm = [i for i, el in enumerate(events[:, 2]) if el == 1001][3:]  # init stim
    frm = events[frm, 0]
    to = events[to, 0]

    raws = [raw.copy().crop(f / fs, t / fs + 1) for f, t in zip(frm, to)]

    raw = raws.pop(0)
    for r in raws:
        raw.append(r)
    del raws

    events, event_id = events_from_annotations(raw)
    events[:, 0] = events[:, 0] - raw.first_samp

    ev = {ev[0]: ev[2] for ev in events}
    data = raw.get_data()
    return data, ev


def run(filename, get_labels=False, eeg_type='', use_artificial_data=False):
    raw = open_raw_file(filename)

    # strip channel names of "." characters
    raw.rename_channels(lambda x: x.strip('.'))

    FS = raw.info['sfreq']
    electrodes = list()

    if eeg_type == M_BRAIN_TRAIN:
        # T9 / Tp9, T10 / Tp10
        electrodes = ['Fp1', 'Fp2', 'Fz', 'F7', 'F8', 'Fc1', 'Fc2', 'Cz',
                      'C3', 'C4', 'T7', 'T8', 'Cpz', 'Cp1', 'Cp2', 'Cp5',
                      'Cp6', 'T9', 'T10', 'Pz', 'P3', 'P4', 'O1', 'O2']
        raw = raw.pick_channels(electrodes)

    electrodes = raw.info['ch_names'].copy()

    if get_labels:
        electrodes.append('trigger')

    info = StreamInfo('BioSemi', 'EEG', len(electrodes), FS, 'float32', 'myuid2424')

    # append some meta-data
    info.desc().append_child_value("manufacturer", "BioSemi")
    channels = info.desc().append_child("channels")

    for c in electrodes:
        channels.append_child("channel") \
            .append_child_value("label", c) \
            .append_child_value("unit", "microvolts") \
            .append_child_value("type", "EEG")

    # next make an outlet; we set the transmission chunk size to 32 samples and
    # the outgoing buffer size to 360 seconds (max.)
    outlet = StreamOutlet(info, 32, 360)
    print("now sending data...")

    if use_artificial_data:
        data = get_artificial_data(24, 2 * 60, FS)
    else:
        if get_labels:
            data, ev = get_data_with_labels(raw)
        else:
            data = raw.get_data()
    stim = 1

    for t in range(np.size(data, axis=1)):
        # make a new random 8-channel sample; this is converted into a
        # pylsl.vectorf (the data type that is expected by push_sample)
        mysample = list(data[:, t])
        if get_labels:
            stim = ev.get(t, stim)
            mysample.append(stim)
        # get a time stamp in seconds (we pretend that our samples are actually
        # 125ms old, e.g., as if coming from some external hardware)
        stamp = local_clock() - 0.125
        # now send it and wait for a bit
        outlet.push_sample(mysample, stamp)
        time.sleep(1 / FS)


if __name__ == '__main__':
    base_dir = "D:/BCI_stuff/databases/"  # MTA TTK
    # base_dir = 'D:/Csabi/'  # Home
    # base_dir = "D:/Users/Csabi/data"  # ITK
    # base_dir = "/home/csabi/databases/"  # linux
    files = [base_dir + "physionet.org/physiobank/database/eegmmidb/S001/S001R03.edf",
             base_dir + "physionet.org/physiobank/database/eegmmidb/S001/S001R04.edf"]

    test_subj = 4
    file = '{}Cybathlon_pilot/pilot{}/rec01.vhdr'.format(base_dir, test_subj)

    run(file, get_labels=False)

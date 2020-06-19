"""Example program to demonstrate how to send a multi-channel time-series
with proper meta-data to LSL."""

import time

import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock

from preprocess.ioprocess import open_raw_file, init_base_config

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

    from_ = [i for i, el in enumerate(events[:, 2]) if el == 1001][3:]  # init stim
    to_ = [i for i, el in enumerate(events[:, 2]) if el == 12][3:]  # end stim
    from_ = events[from_, 0]
    to_ = events[to_, 0]

    raws = [raw.copy().crop(frm / fs, to / fs + 1) for frm, to in zip(from_, to_)]

    raw = raws.pop(0)
    for r in raws:
        raw.append(r)
    del raws

    events, event_id = events_from_annotations(raw)
    events[:, 0] = events[:, 0] - raw.first_samp

    ev = {ev[0]: ev[2] for ev in events}
    data = raw.get_data()
    return data, ev, raw


def run(filename, get_labels=False, eeg_type='', use_artificial_data=False, host='myuid1236', add_extra_data=False):
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

    ch_num = len(electrodes) if not add_extra_data else len(electrodes) + 1
    info = StreamInfo('MNE', 'EEG', ch_num, FS, 'double64', host)

    # append some meta-data
    info.desc().append_child_value("manufacturer", "MNE")
    channels = info.desc().append_child("channels")

    for c in electrodes:
        channels.append_child("channel") \
            .append_child_value("label", c) \
            .append_child_value("unit", "microvolts") \
            .append_child_value("type", "EEG")

    if add_extra_data:
        channels.append_child("channel") \
            .append_child_value("label", "tmp") \
            .append_child_value("unit", "microvolts") \
            .append_child_value("type", "TMP")

    # next make an outlet; we set the transmission chunk size to 32 samples and
    # the outgoing buffer size to 360 seconds (max.)
    outlet = StreamOutlet(info, 32, 360)
    print("now sending data...")

    if use_artificial_data:
        data = get_artificial_data(24, 2 * 60, FS)
    else:
        if get_labels:
            data, ev, _ = get_data_with_labels(raw)
        else:
            data = raw.get_data()

    if add_extra_data:
        d = data
        ch, t = d.shape
        data = np.zeros((ch + 1, t))
        data[:-1, :] = d

    stim = 1
    prevstamp = local_clock() - 0.125 - 1 / FS
    stims = list()
    for t in range(np.size(data, axis=1)):
        # tic = time.time()

        mysample = list(data[:, t])
        if get_labels:
            stim = ev.get(t, stim)
            stims.append(stim)
            mysample.append(stim)
        # get a time stamp in seconds (we pretend that our samples are actually
        # 125ms old, e.g., as if coming from some external hardware)
        stamp = local_clock() - 0.125
        real_sleep_time = stamp - prevstamp
        # print('time spent in sleep in hz: {}'.format(1 / (real_sleep_time)))
        corr = (FS - 1 / real_sleep_time)
        # print('time diff: {} %'.format(real_sleep_time * FS * 100 - 100))
        prevstamp = stamp
        # now send it and wait for a bit
        outlet.push_sample(mysample, stamp)
        time_to_sleep = 1 / (FS + corr)  # max(0, sleep_time - (time.time() - tic))
        # print('time to sleep in hz: {}'.format(1 / (ts)))
        # time.sleep(ts)
        toc = time.clock() + time_to_sleep  # Busy waiting for realtime sleep on Windows...
        while time.clock() < toc:
            pass

    diffstim = set(stims)
    d = {st: 0 for st in diffstim}
    for st in stims:
        d[st] += 1
    print('sent stim:', d)


if __name__ == '__main__':
    from gui_handler import select_file_in_explorer

    base_dir = init_base_config('../')
    file = select_file_in_explorer(base_dir)

    run(file, get_labels=False, add_extra_data=True)

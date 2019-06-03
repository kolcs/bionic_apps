"""Example program to demonstrate how to send a multi-channel time-series
with proper meta-data to LSL."""

import time
from pylsl import StreamInfo, StreamOutlet, local_clock
import numpy as np

from preprocess.ioprocess import open_raw_file

M_BRAIN_TRAIN = 'mBrainTrain'


def run(filename, eeg_type=M_BRAIN_TRAIN):
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

    electrodes = raw.info['ch_names']

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

    data = raw.get_data()
    for t in range(np.size(data, axis=1)):
        # make a new random 8-channel sample; this is converted into a
        # pylsl.vectorf (the data type that is expected by push_sample)
        mysample = data[:, t]
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

    run(files[0])

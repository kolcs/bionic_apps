"""Example program to show how to read a multi-channel time series from LSL."""

from pylsl import StreamInlet, resolve_stream
import numpy as np


def load_electrodes(info):
    electrodes = []
    ch = info.desc().child("channels").child("channel")
    for _ in range(info.channel_count()):
        electrodes.append(ch.child_value("label"))
        ch = ch.next_sibling()
    return electrodes


# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

# get information...
info = inlet.info()
print(info.as_xml())
print(info.nominal_srate())
print(load_electrodes(info))
# exit(0)
timestamps = list()
data = list()

while True:
    sample, timestamp = inlet.pull_sample()
    # print(timestamp, sample)
    timestamps.append(timestamp)
    data.append(sample)
    # print(np.transpose(data).shape)
    print(sample)

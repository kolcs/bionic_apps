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


EXIT = 2
i = 0
# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

# get information...
info = inlet.info()
print(info.as_xml())
print(info.nominal_srate())

data, _ = inlet.pull_sample()
data = np.transpose(np.array(data, ndmin=2))
print(data)
while i < EXIT:
    sample, timestamp = inlet.pull_sample()
    # print(timestamp, sample)
    sample = np.array(sample, ndmin=2)
    sample = np.transpose(sample)
    data = np.append(data, sample, axis=1)
    print(data)
    i += 1

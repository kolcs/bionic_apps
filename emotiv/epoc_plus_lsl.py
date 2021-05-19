# -*- coding: utf8 -*-
#
#  CyKIT  2020.06.05
#  ________________________
#  example_epoc_plus.py
#
#  Written by Warren
#
"""
  CyKIT is needed for this program to run!!!
  CyKit-master\Examples\epoc_plus_lsl.py

  usage:  python.exe .\example_epoc_plus.py

  ( May need to adjust the key below, based on whether
    device is in 14-bit mode or 16-bit mode. )

"""

import os
import sys

print(str(sys.path))
master_parh = r'C:\Users\User\Documents\Rita\1etem\5.szemeszter\CyKit-master'
sys.path.insert(0, master_parh + r'\py3\cyUSB')
sys.path.insert(0, master_parh + r'\py3')

import cyPyWinUSB as hid
import queue
from cyCrypto.Cipher import AES
from cyCrypto import Random

tasks = queue.Queue()
EPOC_PLUS = 'Epoc_plus'

class EEG(object):

    def __init__(self):
        self.hid = None
        self.delimiter = ", "

        devicesUsed = 0

        for device in hid.find_all_hid_devices():
            if device.product_name == 'EEG Signals':
                devicesUsed += 1
                self.hid = device
                self.hid.open()
                self.serial_number = device.serial_number
                device.set_raw_data_handler(self.dataHandler)
        if devicesUsed == 0:
            os._exit(0)
        sn = self.serial_number

        # EPOC+ in 16-bit Mode.
        k = ['\0'] * 16
        k = [sn[-1], sn[-2], sn[-2], sn[-3], sn[-3], sn[-3], sn[-2], sn[-4], sn[-1], sn[-4], sn[-2], sn[-2], sn[-4],
             sn[-4], sn[-2], sn[-1]]

        # EPOC+ in 14-bit Mode.
        # k = [sn[-1],00,sn[-2],21,sn[-3],00,sn[-4],12,sn[-3],00,sn[-2],68,sn[-1],00,sn[-2],88]

        self.key = str(''.join(k))
        self.cipher = AES.new(self.key.encode("utf8"), AES.MODE_ECB)

    def dataHandler(self, data):
        join_data = ''.join(map(chr, data[1:]))
        data = self.cipher.decrypt(bytes(join_data, 'latin-1')[0:32])
        if str(data[1]) == "32":  # No Gyro Data.
            return
        tasks.put(data)

    def convertEPOC_PLUS(self, value_1, value_2):
        edk_value = "%.8f" % (
                    ((int(value_1) * .128205128205129) + 4201.02564096001) + ((int(value_2) - 128) * 32.82051289))
        return edk_value

    def get_data(self):

        data = tasks.get()
        # print(str(data[0])) COUNTER

        try:
            packet_data = ""
            for i in range(2, 16, 2):
                packet_data = packet_data + str(self.convertEPOC_PLUS(str(data[i]), str(data[i + 1]))) + self.delimiter

            for i in range(18, len(data), 2):
                packet_data = packet_data + str(self.convertEPOC_PLUS(str(data[i]), str(data[i + 1]))) + self.delimiter

            packet_data = packet_data[:-len(self.delimiter)]
            return str(packet_data)

        except Exception as exception2:
            print(str(exception2))


import time
import numpy as np
from random import random as rand

from pylsl import StreamInfo, StreamOutlet, local_clock

electrodes = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

#https://labstreaminglayer.readthedocs.io/projects/liblsl/ref/streaminfo.html
info = StreamInfo('Epoc_plus', 'EEG', 14, 128, 'float32', 'EpocPlusEEG_Stream_Rita')
#(Default) Outputs 14 data channels in float format. 128 SPS / 256 SPS (2048 Hz internal)
#Run CyKIT.py, choose MODEL#6 for Epoc+ (default is MODEL#1)

# append some meta-data
info.desc().append_child_value("manufacturer", "Emotiv")
channels = info.desc().append_child("channels")

for c in electrodes:
    channels.append_child("channel") \
        .append_child_value("label", c) \
        .append_child_value("unit", "microvolts") \
        .append_child_value("type", "EEG")

# next make an outlet; we set the transmission chunk size to 32 samples and
# the outgoing buffer size to 360 seconds (max.)
outlet = StreamOutlet(info, 32, 360)

cyHeadset = EEG()

print("now sending data...")
while 1:
    while tasks.empty():
        pass

    mysample = []
    sample = cyHeadset.get_data().split(", ")
    for s in sample:
        mysample.append(float(s))
    #print(mysample)
    # get a time stamp in seconds
    stamp = local_clock()

    outlet.push_sample(mysample, stamp)
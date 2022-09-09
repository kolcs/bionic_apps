"""
Simple Python RDA client for the RDA tcpip interface of the BrainVision Recorder
It reads all the information from the recorded EEG,
prints EEG and marker information to the console and calculates and
prints the average power every second


Brain Products GmbH
Gilching/Freiburg, Germany
www.brainproducts.com

"""

from socket import socket, AF_INET, SOCK_STREAM
from struct import unpack


# Marker class for storing marker information
class Marker:
    def __init__(self):
        self.position = 0
        self.points = 0
        self.channel = -1
        self.type = ""
        self.description = ""


# Helper function for receiving whole message
def recv_data(connection, requested_size):
    return_stream = ''
    while len(return_stream) < requested_size:
        data_bytes = connection.recv(requested_size - len(return_stream))
        if data_bytes == '':
            raise RuntimeError("Connection broken")
        return_stream += data_bytes

    return return_stream


# Helper function for splitting a raw array of
# zero terminated strings (C) into an array of python strings
def split_string(raw):
    string_list = []
    s = ""
    for i in range(len(raw)):
        if raw[i] != '\x00':
            s = s + raw[i]
        else:
            string_list.append(s)
            s = ""

    return string_list


# Helper function for extracting eeg properties from a raw data array
# read from tcpip socket
def get_properties(rawdata):
    # Extract numerical data
    channel_count, sampling_interval = unpack('<Ld', rawdata[:12])

    # Extract resolutions
    resolutions = []
    for c in range(channel_count):
        index = 12 + c * 8
        res_tuple = unpack('<d', rawdata[index:index + 8])
        resolutions.append(res_tuple[0])

    # Extract channel names
    channel_names = split_string(rawdata[12 + 8 * channel_count:])

    return channel_count, sampling_interval, resolutions, channel_names


# Helper function for extracting eeg and marker data from a raw data array
# read from tcpip socket       
def get_data(rawdata, channel_count):
    # Extract numerical data
    block, points, marker_count = unpack('<LLL', rawdata[:12])

    # Extract eeg data as array of floats
    data = []
    for i in range(points * channel_count):
        index = 12 + 4 * i
        value = unpack('<f', rawdata[index:index + 4])
        data.append(value[0])

    # Extract markers
    markers = []
    index = 12 + 4 * points * channel_count
    for m in range(marker_count):
        marker_size = unpack('<L', rawdata[index:index + 4])

        ma = Marker()
        ma.position, ma.points, ma.channel = unpack('<LLl', rawdata[index + 4:index + 16])
        type_desc = split_string(rawdata[index + 16:index + marker_size[0]])
        ma.type = type_desc[0]
        ma.description = type_desc[1]

        markers.append(ma)
        index = index + marker_size[0]

    return block, points, marker_count, data, markers


def main_demo():
    # Create a tcpip socket
    con = socket(AF_INET, SOCK_STREAM)
    # Connect to recorder host via 32Bit RDA-port
    # adapt to your host, if recorder is not running on local machine
    # change port to 51234 to connect to 16Bit RDA-port
    con.connect(("localhost", 51244))
    channel_count, sampling_interval, resolutions, channel_names = None, None, None, None

    # Flag for main loop
    finish = False

    # data buffer for calculation, empty in beginning
    data1s = []

    # block counter to check overflows of tcpip buffer
    last_block = -1

    # Main Loop #
    while not finish:

        # Get message header as raw array of chars
        rawhdr = recv_data(con, 24)

        # Split array into useful information id1 to id4 are constants
        (id1, id2, id3, id4, msg_size, msg_type) = unpack('<llllLL', rawhdr)

        # Get data part of message, which is of variable size
        rawdata = recv_data(con, msg_size - 24)

        # Perform action dependent on the message type
        if msg_type == 1:
            # Start message, extract eeg properties and display them
            channel_count, sampling_interval, resolutions, channel_names = get_properties(rawdata)
            # reset block counter
            last_block = -1

            print("Start")
            print("Number of channels: " + str(channel_count))
            print("Sampling interval: " + str(sampling_interval))
            print("Resolutions: " + str(resolutions))
            print("Channel Names: " + str(channel_names))

        elif msg_type == 4:
            # Data message, extract data and markers
            block, points, marker_count, data, markers = get_data(rawdata, channel_count)

            # Check for overflow
            if last_block != -1 and block > last_block + 1:
                print("*** Overflow with " + str(block - last_block) + " datablocks ***")
            last_block = block

            # Print markers, if there are some in actual block
            if marker_count > 0:
                for m in range(marker_count):
                    print("Marker " + markers[m].description + " of type " + markers[m].type)

            # Put data at the end of actual buffer
            data1s.extend(data)

            # If more than 1s of data is collected, calculate average power, print it and reset data buffer
            if len(data1s) > channel_count * 1000000 / sampling_interval:
                index = int(len(data1s) - channel_count * 1000000 / sampling_interval)
                data1s = data1s[index:]

                avg = 0
                # Do not forget to respect the resolution !!!
                for i in range(len(data1s)):
                    avg = avg + data1s[i] * data1s[i] * resolutions[i % channel_count] * resolutions[i % channel_count]

                avg = avg / len(data1s)
                print("Average power: " + str(avg))

                data1s = []

        elif msg_type == 3:
            # Stop message, terminate program
            print("Stop")
            finish = True

    # Close tcpip connection
    con.close()


if __name__ == '__main__':
    main_demo()

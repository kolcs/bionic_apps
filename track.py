import json
import numpy as np

TRACK_LENGTH = 500  # m

SEGMENTS = 'segments'
LENGTH = 'length'
RADIUS = 'radius'
ANGLE = 'angle'
DARK_END = 'potentialDarkEnd'
DARK_START = 'potentialDarkStart'
DARK_LENGTH = 'darkLength'
COUNT = 'count'

LEFT_ZONE = 'Left zone'
RIGHT_ZONE = 'Right zone'
LIGHT_ZONE = 'Light zone'
STRAIGHT_ZONE = 'Straight zone'


def calculate_arc_length(angle, radius):
    return np.abs(radius * np.pi / 180 * angle)


def get_track_length(length, angle, radius):
    if radius == 0:
        return length
    return calculate_arc_length(angle, radius)


class TrackGenerator:

    def __init__(self):
        self._track = {SEGMENTS: []}
        self._stat = {LEFT_ZONE: {COUNT: 0, LENGTH: 0},
                      RIGHT_ZONE: {COUNT: 0, LENGTH: 0},
                      LIGHT_ZONE: {COUNT: 0, LENGTH: 0},
                      STRAIGHT_ZONE: {COUNT: 0, LENGTH: 0}}
        self.total_length = 0

    def _update_stat(self, zone, length):
        self._stat[zone][COUNT] += 1
        self._stat[zone][LENGTH] += length

    def calculate_statistics(self):
        for element in self._track[SEGMENTS]:
            length = get_track_length(element[LENGTH], element[ANGLE], element[RADIUS])
            self.total_length += length
            if element[DARK_END] > 0:
                self._update_stat(LIGHT_ZONE, length)
            elif element[RADIUS] < 0:
                self._update_stat(RIGHT_ZONE, length)
            elif element[RADIUS] > 0:
                self._update_stat(LEFT_ZONE, length)
            else:
                self._update_stat(STRAIGHT_ZONE, length)

    def print_stat(self):
        for zone, data in self._stat.items():
            print(zone)
            print('\t{}: {}'.format(COUNT, data[COUNT]))
            print('\t{}: {}'.format(LENGTH, data[LENGTH]))
        print('Total length: {}'.format(self.total_length))

    def load_from_file(self, filename):
        with open(filename) as json_file:
            self._track = json.load(json_file)
        self.calculate_statistics()


if __name__ == '__main__':
    gen = TrackGenerator()
    gen.load_from_file(r'D:\Users\Csabi\Desktop\BrainDriver_V1.6.1\trackData.json')
    gen.print_stat()

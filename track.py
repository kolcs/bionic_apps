import json
import numpy as np

TRACK_LENGTH = 500  # m
RADIUS_VALUE = 17.829999923706056
DARK_RND = 4

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


def _get_segment(length=.0, radius=.0, angle=.0, dark_start=.0, dark_end=.0, dark_length=.0):
    assert dark_start <= length, 'potentialDarkStart should be smaller than length'
    assert dark_start <= dark_end, 'potentialDarkStart should be smaller than potentialDarkEnd'
    assert dark_end - dark_start >= dark_length, 'darkLength is longer, than the allowed interval'
    return {LENGTH: length,
            RADIUS: radius,
            ANGLE: angle,
            DARK_START: dark_start,
            DARK_END: dark_end,
            DARK_LENGTH: dark_length}


def _get_straight_segment(length):
    return _get_segment(length)


def _get_left_segment(angle):
    return _get_segment(radius=RADIUS_VALUE, angle=angle)


def _get_right_segment(angle):
    return _get_segment(radius=-RADIUS_VALUE, angle=angle)


def _get_dark_segment(length):
    dark_start = np.random.randint(0, DARK_RND + 1, dtype=float)
    dark_end = np.random.randint(length - DARK_RND, length + 1, dtype=float)
    dark_length = dark_end - dark_start
    dark_length = np.random.randint(dark_length - DARK_RND, dark_length + 1, dtype=float)
    return _get_segment(length,
                        dark_start=dark_start,
                        dark_end=dark_end,
                        dark_length=dark_length)


class TrackGenerator:

    def __init__(self):
        self._track = {SEGMENTS: []}
        self._init_stat()

    def _init_stat(self):
        self._stat = {LEFT_ZONE: {COUNT: 0, LENGTH: 0},
                      RIGHT_ZONE: {COUNT: 0, LENGTH: 0},
                      LIGHT_ZONE: {COUNT: 0, LENGTH: 0},
                      STRAIGHT_ZONE: {COUNT: 0, LENGTH: 0}}
        self.total_length = 0

    def _update_stat(self, zone, length):
        self._stat[zone][COUNT] += 1
        self._stat[zone][LENGTH] += length

    def calculate_statistics(self):
        self._init_stat()
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

    def generate(self):
        self._track[SEGMENTS].append(_get_segment(10))


if __name__ == '__main__':
    gen = TrackGenerator()
    gen.load_from_file(r'D:\Users\Csabi\Desktop\BrainDriver_V1.6.1\trackData.json')
    gen.print_stat()

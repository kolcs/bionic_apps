from pathlib import Path
from sys import argv

import numpy as np

from bionic_apps.utils import save_to_json

TRACK_LENGTH = 500.0  # m
RADIUS_VALUE = 17.829999923706056
DARK_RND = 6
LENGTH_RND = 2
INIT_END_ZONE = 10

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

FILE_NAME = 'trackData.json'


def calculate_arc_length(angle, radius):
    return np.abs(radius * np.pi / 180.0 * angle)


def calculate_arc_angle(length, radius):
    return np.abs(length * 180.0 / np.pi / radius)


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


def _get_left_segment(length):
    angle = calculate_arc_angle(length, RADIUS_VALUE)
    return _get_segment(radius=RADIUS_VALUE, angle=angle)


def _get_right_segment(length):
    angle = calculate_arc_angle(length, -RADIUS_VALUE)
    return _get_segment(radius=-RADIUS_VALUE, angle=angle)


def _get_light_segment(length):
    dark_start = np.random.randint(0, DARK_RND + 1)
    dark_end = np.random.randint(length - DARK_RND, length + 1)
    dark_length = dark_end - dark_start
    dark_length = np.random.randint(dark_length - DARK_RND, dark_length + 1)
    return _get_segment(length=length,
                        dark_start=dark_start,
                        dark_end=dark_end,
                        dark_length=dark_length)


def _out_of_repetition_limit(sequence, limit=2):
    prev_seq = str()
    seq_num = 0
    for seq in sequence:
        if seq == prev_seq:
            seq_num += 1
        else:
            seq_num = 0
        if seq_num == limit:
            return True
        prev_seq = seq
    return False


class TrackGenerator:

    def __init__(self, left_zone=4, right_zone=4, light_on=4, straight=4):
        self._track = {SEGMENTS: []}
        self._zone_numbers = {
            LEFT_ZONE: left_zone,
            RIGHT_ZONE: right_zone,
            LIGHT_ZONE: light_on,
            STRAIGHT_ZONE: straight
        }
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
        self.calculate_statistics()
        for zone, data in self._stat.items():
            print(zone)
            print('\t{}: {}'.format(COUNT, data[COUNT]))
            print('\t{}: {}'.format(LENGTH, data[LENGTH]))
        print('Total length: {}\n'.format(self.total_length))

    def _generate_turn_seqs(self):
        turns = [LEFT_ZONE] * self._zone_numbers[LEFT_ZONE] + [RIGHT_ZONE] * self._zone_numbers[RIGHT_ZONE]
        while _out_of_repetition_limit(turns):
            np.random.shuffle(turns)
        return turns

    def _generate_straight_seqs(self):
        straight = [LIGHT_ZONE] * self._zone_numbers[LIGHT_ZONE] + [STRAIGHT_ZONE] * self._zone_numbers[STRAIGHT_ZONE]
        np.random.shuffle(straight)
        return straight

    def _add_segment_to_track(self, segment):
        self._track[SEGMENTS].append(segment)

    def _get_zone_length(self, zone, shorten=.0):
        self.calculate_statistics()
        zone_full_length = TRACK_LENGTH / 4 - shorten
        if self._stat[zone][COUNT] + 1 < self._zone_numbers[zone]:  # one before last
            return 2 * LENGTH_RND * np.random.random_sample() - LENGTH_RND + \
                   zone_full_length / self._zone_numbers[zone]
        return zone_full_length - self._stat[zone][LENGTH]  # last

    def generate(self):
        turns = self._generate_turn_seqs()
        straights = self._generate_straight_seqs()

        last_is_straight = False

        while len(turns) > 0:
            turn = turns.pop(0)
            if turn == LEFT_ZONE:
                length = self._get_zone_length(LEFT_ZONE)
                self._add_segment_to_track(_get_left_segment(length))
            elif turn == RIGHT_ZONE:
                length = self._get_zone_length(RIGHT_ZONE)
                self._add_segment_to_track(_get_right_segment(length))
            straight = straights.pop(0)
            if straight == LIGHT_ZONE:
                length = self._get_zone_length(LIGHT_ZONE)
                self._add_segment_to_track(_get_light_segment(length))
            elif straight == STRAIGHT_ZONE:
                if len(straights) == 0:
                    length = self._get_zone_length(STRAIGHT_ZONE, INIT_END_ZONE)
                    last_is_straight = True
                else:
                    length = self._get_zone_length(STRAIGHT_ZONE, INIT_END_ZONE * 2)
                self._add_segment_to_track(_get_straight_segment(length))

        self._track[SEGMENTS].insert(0, _get_straight_segment(INIT_END_ZONE))
        if not last_is_straight:
            self._add_segment_to_track(_get_straight_segment(INIT_END_ZONE))

    def save_to_path(self, path='./'):
        file = Path(path).joinpath(FILE_NAME)
        save_to_json(file, self._track)
        print('Track generated on path: {} with filename {}'.format(file.resolve().parent, file.name))


if __name__ == '__main__':
    gen = TrackGenerator()
    # gen.load_from_file()
    gen.generate()
    gen.print_stat()

    path = argv[1] if len(argv) == 2 else '.'
    gen.save_to_path(path)

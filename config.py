"""
Configuration for databases
"""
# Record types:
IMAGINED_MOVEMENT = "imagined"
REAL_MOVEMENT = "real"
BASELINE = 'baseline'

# Task types:
EYE_OPEN = 'eye open'
EYE_CLOSED = 'eye closed'
LEFT_HAND = 'left hand'
RIGHT_HAND = 'right hand'
BOTH_HANDS = 'both hands'
LEFT_LEG = 'left_leg'
RIGHT_LEG = 'right_leg'
BOTH_LEGS = 'both legs'
REST = 'rest'

SUBJECT = 'subject'

DATA_START_AFTER = 1  # sec
DATA_END_BEFORE = 0.5  # sec

RECORD_TO_NUM = {key: value for value, key in enumerate([IMAGINED_MOVEMENT, REAL_MOVEMENT, BASELINE])}
TASK_TO_NUM = {key: value for value, key in enumerate(
    [EYE_OPEN, EYE_CLOSED, LEFT_HAND, RIGHT_HAND, BOTH_HANDS, LEFT_LEG, RIGHT_LEG, BOTH_LEGS, REST])}
ONE_HOT_LIST = [RECORD_TO_NUM, TASK_TO_NUM]

DATA = 'data'
DATA_SHAPE = 'data shape'
RECORD_TYPE_LABEL = 'record type label'
TASK_LABEL = 'task label'

TRAIN = 'train'
VALIDATION = 'validation'
TEST = 'test'

DIR_TRAIN = 'train/'
DIR_VALIDATION = 'validation/'
DIR_TEST = 'test/'
DIR_TF_RECORDS = 'tfRecords/'
F_EXT_TF_RECORD = '.tfrecord'


class SourceDB:
    DIR = ""
    CHANNEL_NUM = int()
    FS = int()
    DB_EXT = ''

    TRIGGER_TYPE_CONVERTER = dict()
    TRIGGER_TASK_CONVERTER = dict()

    DROP_SUBJECTS = list()

    def convert_type(self, record_number):
        return self.TRIGGER_TYPE_CONVERTER.get(record_number)

    def convert_task(self, record_number):
        return self.TRIGGER_TASK_CONVERTER.get(record_number)


class Physionet(SourceDB):
    DIR = "physionet.org/"
    CHANNEL_NUM = 64
    FS = 160
    DB_EXT = '.edf'

    TASK_EYE_OPEN = {EYE_OPEN: 1}
    TASK_EYE_CLOSED = {EYE_CLOSED: 1}

    TASK_12 = {
        REST: 1,
        LEFT_HAND: 2,
        RIGHT_HAND: 3
    }

    TASK_34 = {
        REST: 1,
        BOTH_HANDS: 2,
        BOTH_LEGS: 3
    }

    TRIGGER_TYPE_CONVERTER = {
        1: BASELINE,
        2: BASELINE,
        3: REAL_MOVEMENT,
        4: IMAGINED_MOVEMENT,
        5: REAL_MOVEMENT,
        6: IMAGINED_MOVEMENT,
        7: REAL_MOVEMENT,
        8: IMAGINED_MOVEMENT,
        9: REAL_MOVEMENT,
        10: IMAGINED_MOVEMENT,
        11: REAL_MOVEMENT,
        12: IMAGINED_MOVEMENT,
        13: REAL_MOVEMENT,
        14: IMAGINED_MOVEMENT
    }

    TRIGGER_TASK_CONVERTER = {  # rec_num : {taskID: task}
        1: TASK_EYE_OPEN,
        2: TASK_EYE_CLOSED,
        3: TASK_12,
        4: TASK_12,
        5: TASK_34,
        6: TASK_34,
        7: TASK_12,
        8: TASK_12,
        9: TASK_34,
        10: TASK_34,
        11: TASK_12,
        12: TASK_12,
        13: TASK_34,
        14: TASK_34
    }

    TRIGGER_TASK_LIST = [
        (3, 4, 7, 8, 11, 12),
        (5, 6, 9, 10, 13, 14)
    ]

    MAX_DURATION = 4  # seconds --> creating strictly formatted data window

    """
       DROP_SUBJECTS:
    
       list of subjects, whos records are corrupted
       89 - wrong baseline session(T0 with T1)
       88, 92, 100 - wrong intervals (1,375, 5,125) and freq 128Hz instead of 160Hz
       """
    DROP_SUBJECTS = [89, 88, 92, 100]

    # # source:
    # list(range(35, 42)) + list(range(46, 53)) + list(range(57, 64)) + list(range(4, 7)) + list(
    # range(14, 19)) + list(range(23, 32)) + [34, 42, 45, 53, 44, 54, 56, 64] + list(range(67, 76)) + list(
    # range(80, 85)) + list(range(92, 95)) + [104]
    CHANNEL_TRANSFORMATION = [35, 36, 37, 38, 39, 40, 41, 46, 47, 48, 49, 50, 51, 52, 57, 58, 59, 60, 61, 62, 63, 4, 5,
                              6, 14, 15, 16, 17, 18, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 42, 45, 53, 44, 54, 56, 64,
                              67, 68, 69, 70, 71, 72, 73, 74, 75, 80, 81, 82, 83, 84, 92, 93, 94, 104]

"""
Configuration for databases
"""
# Record types:
IMAGINED_MOVEMENT = "imagined"
REAL_MOVEMENT = "real"
BASELINE = 'baseline'

# Task types: if you modify it you have to rerun the preprocess!
EYE_OPEN = 'eye open'
EYE_CLOSED = 'eye closed'
LEFT_HAND = 'left hand'
RIGHT_HAND = 'right hand'
BOTH_HANDS = 'both hands'
LEFT_LEG = 'left leg'
RIGHT_LEG = 'right leg'
BOTH_LEGS = 'both legs'
REST = 'rest'

SUBJECT = 'subject'

# control commands
TURN_LEFT = 'turn left'
TURN_RIGHT = 'turn right'
LIGHT_ON = 'light on'
GO_STRAIGHT = 'go straight'

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

DIR_FEATURE_DB = 'tmp/'


# class SourceDB: # JUST a template!!!
#     DIR = ""
#     CHANNEL_NUM = int()
#     FS = int()
#     DB_EXT = ''
#
#     TRIGGER_CONV_REC_TO_TYPE = dict()
#     TRIGGER_CONV_REC_TO_TASK = dict()
#     or
#     TRIGGER_TASK_CONVERTER
#
#     DROP_SUBJECTS = list()

class PilotBo:
    DIR = "PilotBo/"
    TRIGGER_TASK_CONVERTER = {  # imagined
        REST: 1,
        RIGHT_HAND: 5,
        LEFT_HAND: 7,
        # RIGHT_LEG: 9,
        # LEFT_LEG: 11
        # BOTH_HANDS: 9,
        BOTH_LEGS: 11
    }

    COMMAND_CONV = {
        REST: GO_STRAIGHT,
        RIGHT_HAND: TURN_RIGHT,
        LEFT_HAND: TURN_LEFT,
        BOTH_LEGS: LIGHT_ON
    }

    TRIGGER_EVENT_ID = {'Stimulus/S ' + (2 - len(str(i + 1))) * ' ' + str(i + 1): i + 1 for i in range(16)}
    DROP_SUBJECTS = []


class PilotCs:
    DIR = "PilotCs/"
    TRIGGER_TASK_CONVERTER = {  # imagined
        REST: 1,
        RIGHT_HAND: 5,
        LEFT_HAND: 7,
        # RIGHT_LEG: 9,
        # LEFT_LEG: 11
        # BOTH_HANDS: 9,
        BOTH_LEGS: 11
    }

    COMMAND_CONV = {
        REST: GO_STRAIGHT,
        RIGHT_HAND: TURN_RIGHT,
        LEFT_HAND: TURN_LEFT,
        BOTH_LEGS: LIGHT_ON
    }

    TRIGGER_EVENT_ID = {'Stimulus/S ' + (2 - len(str(i + 1))) * ' ' + str(i + 1): i + 1 for i in range(16)}
    DROP_SUBJECTS = []


class GameDB:
    DIR = "Game/"
    TRIGGER_TASK_CONVERTER = {  # imagined
        REST: 1,
        RIGHT_HAND: 5,
        LEFT_HAND: 7,
        # RIGHT_LEG: 9,
        # LEFT_LEG: 11
        # BOTH_HANDS: 9,
        BOTH_LEGS: 11
    }

    COMMAND_CONV = {
        REST: GO_STRAIGHT,
        RIGHT_HAND: TURN_RIGHT,
        LEFT_HAND: TURN_LEFT,
        BOTH_LEGS: LIGHT_ON
    }

    TRIGGER_EVENT_ID = {'Stimulus/S ' + (2 - len(str(i + 1))) * ' ' + str(i + 1): i + 1 for i in range(16)}
    DROP_SUBJECTS = []


class PilotDB:
    DIR = "Cybathlon_pilot/paradigmA/"
    CHANNEL_NUM = 63
    FS = 500
    DB_EXT = '.vhdr'
    SUBJECT_NUM = 4
    FILE_PATH = 'pilot{subj}/rec{rec}.vhdr'

    TRIGGER_TASK_CONVERTER = {  # imagined
        REST: 1,
        # EYE_OPEN: 2,
        # EYE_CLOSED: 3,
        RIGHT_HAND: 5,
        LEFT_HAND: 7,
        RIGHT_LEG: 9,
        LEFT_LEG: 11
    }

    TRIGGER_EVENT_ID = {'Stimulus/S ' + (2 - len(str(i + 1))) * ' ' + str(i + 1): i + 1 for i in range(16)}

    DROP_SUBJECTS = []  # [] or [1, 4] or [2, 3]


class PilotDB_ParadigmB:
    DIR = "Cybathlon_pilot/paradigmB/"
    CHANNEL_NUM = 63
    FS = 500
    DB_EXT = '.vhdr'
    SUBJECT_NUM = 4
    FILE_PATH = 'pilot{subj}/rec{rec}.vhdr'

    TRIGGER_TASK_CONVERTER = {  # imagined
        REST: 1,
        # EYE_OPEN: 2,
        # EYE_CLOSED: 3,
        RIGHT_HAND: 5,
        LEFT_HAND: 7,
        BOTH_HANDS: 9,
        BOTH_LEGS: 11
    }

    TRIGGER_EVENT_ID = {'Stimulus/S ' + (2 - len(str(i + 1))) * ' ' + str(i + 1): i + 1 for i in range(16)}

    DROP_SUBJECTS = []  # [] or [1, 3] or [2, 4]


class TTK_DB:
    DIR = "TTK/"
    CHANNEL_NUM = 63
    FS = 500
    DB_EXT = '.vhdr'
    SUBJECT_NUM = 25
    FILE_PATH = 'subject{subj}/rec{rec}.vhdr'

    TRIGGER_TASK_CONVERTER = {  # imagined
        REST: 1,
        # EYE_OPEN: 2,
        # EYE_CLOSED: 3,
        RIGHT_HAND: 5,
        LEFT_HAND: 7,
        RIGHT_LEG: 9,
        LEFT_LEG: 11
    }

    TRIGGER_EVENT_ID = {'Stimulus/S ' + (2 - len(str(i + 1))) * ' ' + str(i + 1): i + 1 for i in range(16)}

    DROP_SUBJECTS = [1, 9, 17]


class Physionet:
    # What is FP region in channels???
    DIR = "physionet.org/"
    CHANNEL_NUM = 64
    FS = 160
    DB_EXT = '.edf'
    SUBJECT_NUM = 109
    FILE_PATH = 'physiobank/database/eegmmidb/S{subj}/S{subj}R{rec}.edf'

    TRIGGER_EVENT_ID = {'T{}'.format(i): i + 1 for i in range(3)}

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

    TRIGGER_CONV_REC_TO_TYPE = {
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

    TYPE_TO_REC = {
        BASELINE: [1, 2],
        REAL_MOVEMENT: [i for i in range(3, 15, 2)],
        IMAGINED_MOVEMENT: [i for i in range(4, 15, 2)]
    }

    TRIGGER_CONV_REC_TO_TASK = {  # rec_num : {taskID: task}
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

    TASK_TO_REC = {  # same trigger in leg-hand and left-right
        REST: [i for i in range(4, 15, 2)],
        LEFT_HAND: [i for i in range(4, 15, 4)],
        RIGHT_HAND: [i for i in range(4, 15, 4)],
        BOTH_HANDS: [i for i in range(6, 15, 4)],
        BOTH_LEGS: [i for i in range(6, 15, 4)]
    }

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

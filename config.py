"""
Configuration for databases
"""
from enum import Enum

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
ACTIVE = 'active'
CALM = 'calm'
TONGUE = 'tongue'

SUBJECT = 'subject'


class ControlCommand(Enum):
    LEFT = 1
    RIGHT = 3
    HEADLIGHT = 2
    STRAIGHT = 0


DIR_FEATURE_DB = 'tmp/'

# Record types:
IMAGINED_MOVEMENT = "imagined"
REAL_MOVEMENT = "real"
BASELINE = 'baseline'


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


class GameDB:
    DIR = "Game/mixed"
    FILE_PATH = 'subject{subj}/rec{rec}.vhdr'

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
        REST: ControlCommand.STRAIGHT,
        RIGHT_HAND: ControlCommand.RIGHT,
        LEFT_HAND: ControlCommand.LEFT,
        BOTH_LEGS: ControlCommand.HEADLIGHT
    }

    TRIGGER_EVENT_ID = {'Stimulus/S ' + (2 - len(str(i + 1))) * ' ' + str(i + 1): i + 1 for i in range(16)}

    DROP_SUBJECTS = []


class Game_ParadigmC:
    DIR = "Game/paradigmC/"
    FILE_PATH = 'subject{subj}/rec{rec}.vhdr'

    TRIGGER_TASK_CONVERTER = {  # imagined
        # REST: 1,
        # EYE_OPEN: 2,
        # EYE_CLOSED: 3,
        RIGHT_HAND: 5,
        LEFT_HAND: 7,
        CALM: 9,
        BOTH_LEGS: 11
    }

    COMMAND_CONV = {
        CALM: ControlCommand.STRAIGHT,
        RIGHT_HAND: ControlCommand.RIGHT,
        LEFT_HAND: ControlCommand.LEFT,
        BOTH_LEGS: ControlCommand.HEADLIGHT
    }

    TRIGGER_EVENT_ID = {'Stimulus/S ' + (2 - len(str(i + 1))) * ' ' + str(i + 1): i + 1 for i in range(16)}

    DROP_SUBJECTS = []


class ParadigmC:
    DIR = "ParC/"
    FILE_PATH = 'subject{subj}/rec{rec}.vhdr'

    TRIGGER_TASK_CONVERTER = {  # imagined
        # REST: 1,
        # EYE_OPEN: 2,
        # EYE_CLOSED: 3,
        RIGHT_HAND: 5,
        LEFT_HAND: 7,
        CALM: 9,
        BOTH_LEGS: 11
    }

    COMMAND_CONV = {
        CALM: ControlCommand.STRAIGHT,
        RIGHT_HAND: ControlCommand.RIGHT,
        LEFT_HAND: ControlCommand.LEFT,
        BOTH_LEGS: ControlCommand.HEADLIGHT
    }

    TRIGGER_EVENT_ID = {'Stimulus/S ' + (2 - len(str(i + 1))) * ' ' + str(i + 1): i + 1 for i in range(16)}

    DROP_SUBJECTS = []


class Game_ParadigmD:
    DIR = "Game/paradigmD/"
    FILE_PATH = 'subject{subj}/rec{rec}.vhdr'

    TRIGGER_TASK_CONVERTER = {  # imagined
        # REST: 1,
        # EYE_OPEN: 2,
        # EYE_CLOSED: 3,
        ACTIVE + '1': 5,
        ACTIVE + '2': 9,
        CALM + '1': 7,
        CALM + '2': 11
    }

    TRIGGER_EVENT_ID = {'Stimulus/S ' + (2 - len(str(i + 1))) * ' ' + str(i + 1): i + 1 for i in range(16)}

    DROP_SUBJECTS = []


class PilotDB_ParadigmA:
    DIR = "Cybathlon_pilot/paradigmA/"
    FILE_PATH = 'pilot{subj}/rec{rec}.vhdr'

    TRIGGER_TASK_CONVERTER = {  # imagined
        # REST: 1,
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
    FILE_PATH = 'pilot{subj}/rec{rec}.vhdr'

    TRIGGER_TASK_CONVERTER = {  # imagined
        # REST: 1,
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
    FILE_PATH = 'subject{subj}/rec{rec}.vhdr'

    TRIGGER_TASK_CONVERTER = {  # imagined
        # REST: 1,
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
        # REST: [i for i in range(4, 15, 2)],
        LEFT_HAND: [i for i in range(4, 15, 4)],
        RIGHT_HAND: [i for i in range(4, 15, 4)],
        BOTH_HANDS: [i for i in range(6, 15, 4)],
        BOTH_LEGS: [i for i in range(6, 15, 4)]
    }

    MAX_DURATION = 4  # seconds --> creating strictly formatted data window

    """
    DROP_SUBJECTS: list of subjects, whose records are corrupted
        89 - wrong baseline session (T0 with T1)
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


class BciCompIV2a:
    DIR = "BCI_comp/4/2a"
    FILE_PATH = 'A{subj}T_raw.fif'  # .gdf
    SUBJECT_NUM = 9

    TRIGGER_TASK_CONVERTER = {  # imagined
        LEFT_HAND: 4,
        RIGHT_HAND: 5,
        BOTH_LEGS: 6,
        TONGUE: 7
    }

    TRIGGER_EVENT_ID = {str(el): i + 1 for i, el in enumerate([276, 277, 768, 769, 770, 771, 772, 783,
                                                               1023, 1072, 32766])}

    DROP_SUBJECTS = []


class BciCompIV2b:
    DIR = "BCI_comp/4/2b"
    FILE_PATH = 'B{subj}{rec}T_raw.fif'  # .gdf
    SUBJECT_NUM = 27  # actually 9 but one subject has 3 sessions: 1,2 - no feedback, 3 - feedback

    TRIGGER_TASK_CONVERTER = {  # imagined
        LEFT_HAND: 4,
        RIGHT_HAND: 5,
    }

    TRIGGER_EVENT_ID = {str(el): i + 1 for i, el in enumerate([276, 277, 768, 769, 770, 781, 783, 1023,
                                                               1077, 1078, 1079, 1081, 32766])}

    DROP_SUBJECTS = []


class BciCompIV1:
    """
    In this dataset there are only 2 classes out of left, right, foot at each subject.
    It is not suggested to train it in a cross subject fashion...
    """
    DIR = "BCI_comp/4/1"
    FILE_PATH = 'calib_ds_subj{subj}_raw.fif'  # .gdf
    SUBJECT_NUM = 7

    TRIGGER_TASK_CONVERTER = {  # imagined
        'class1': 1,
        'class2': 2,
    }

    TRIGGER_EVENT_ID = 'auto'

    DROP_SUBJECTS = []

from .defaults import LEFT_HAND, RIGHT_HAND, BOTH_HANDS, BOTH_LEGS, EYE_OPEN, EYE_CLOSED, \
    REST, TONGUE, BASELINE, REAL_MOVEMENT, IMAGINED_MOVEMENT


class Physionet:

    def __init__(self, config_ver=-1.):
        self.DIR = "physionet.org/"
        self.SUBJECT_NUM = 109

        self.CONFIG_VER = 1.1 if config_ver == -1. else config_ver
        if self.CONFIG_VER >= 1:
            self.FILE_PATH = 'S{subj}/S{subj}R{rec}_raw.fif'
            self.SUBJECT_EXP = {subj: (subj,) for subj in range(1, self.SUBJECT_NUM + 1)}

            self.TRIGGER_TASK_CONVERTER = {  # imagined
                # REST: 1,
                LEFT_HAND: 2,
                RIGHT_HAND: 3,
                BOTH_HANDS: 4,
                BOTH_LEGS: 5
            }

            self.TRIGGER_EVENT_ID = {'T{}'.format(i): i + 1 for i in range(5)}
            """
            DROP_SUBJECTS: list of subjects, whose records are corrupted
                89 - wrong baseline session (T0 with T1) 
                88, 92, 100 - wrong intervals (1,375, 5,125) and freq 128Hz instead of 160Hz
            """
            self.DROP_SUBJECTS = [89, 88, 92, 100]  # todo: remove 89?

        else:
            self.FILE_PATH = 'physiobank/database/eegmmidb/S{subj}/S{subj}R{rec}.edf'
            self.TRIGGER_EVENT_ID = {'T{}'.format(i): i + 1 for i in range(3)}

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

            self.TRIGGER_CONV_REC_TO_TYPE = {
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

            self.TYPE_TO_REC = {
                BASELINE: [1, 2],
                REAL_MOVEMENT: [i for i in range(3, 15, 2)],
                IMAGINED_MOVEMENT: [i for i in range(4, 15, 2)]
            }

            self.TRIGGER_CONV_REC_TO_TASK = {  # rec_num : {taskID: task}
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

            self.TASK_TO_REC = {  # same trigger in leg-hand and left-right
                # REST: [i for i in range(4, 15, 2)],
                LEFT_HAND: [i for i in range(4, 15, 4)],
                RIGHT_HAND: [i for i in range(4, 15, 4)],
                BOTH_HANDS: [i for i in range(6, 15, 4)],
                BOTH_LEGS: [i for i in range(6, 15, 4)]
            }

            self.MAX_DURATION = 4  # seconds --> creating strictly formatted data window

            """
            DROP_SUBJECTS: list of subjects, whose records are corrupted
                89 - wrong baseline session (T0 with T1)
                88, 92, 100 - wrong intervals (1,375, 5,125) and freq 128Hz instead of 160Hz
            """
            self.DROP_SUBJECTS = [89, 88, 92, 100]

            # # source:
            # list(range(35, 42)) + list(range(46, 53)) + list(range(57, 64)) + list(range(4, 7)) + list(
            # range(14, 19)) + list(range(23, 32)) + [34, 42, 45, 53, 44, 54, 56, 64] + list(range(67, 76)) + list(
            # range(80, 85)) + list(range(92, 95)) + [104]
            self.CHANNEL_TRANSFORMATION = [35, 36, 37, 38, 39, 40, 41, 46, 47, 48, 49, 50, 51, 52, 57, 58, 59, 60, 61,
                                           62, 63, 4, 5, 6, 14, 15, 16, 17, 18, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34,
                                           42, 45, 53, 44, 54, 56, 64, 67, 68, 69, 70, 71, 72, 73, 74, 75, 80, 81, 82,
                                           83, 84, 92, 93, 94, 104]


class BciCompIV2a:

    def __init__(self, config_ver=-1):
        self.DIR = "BCI_comp/4/2a"
        self.CONFIG_VER = 1.1 if config_ver == -1. else config_ver

        self.FILE_PATH = 'S{subj}/S{subj}R{rec}_raw.fif'
        self.SUBJECT_NUM = 2 * 9
        self.SUBJECT_EXP = {s + 1: [s * 2 + 1, (s + 1) * 2] for s in range(9)}  # must be sorted!

        self.TRIGGER_TASK_CONVERTER = {  # imagined
            LEFT_HAND: 4,
            RIGHT_HAND: 5,
            BOTH_LEGS: 6,
            TONGUE: 7
        }

        self.TRIGGER_EVENT_ID = {str(el): i + 1 for i, el in enumerate([276, 277, 768, 769, 770, 771, 772, 783,
                                                                        1023, 1072, 32766])}

        self.DROP_SUBJECTS = []


class BciCompIV2b:

    def __init__(self, config_ver=-1):
        self.DIR = "BCI_comp/4/2b"
        self.CONFIG_VER = 1.1 if config_ver == -1. else config_ver

        self.FILE_PATH = 'S{subj}/S{subj}{rec}_raw.fif'
        self.SUBJECT_NUM = 5 * 9
        self.SUBJECT_EXP = {s + 1: [s * 5 + i for i in range(1, 6)] for s in range(9)}  # must be sorted!

        self.TRIGGER_TASK_CONVERTER = {  # imagined
            LEFT_HAND: 4,
            RIGHT_HAND: 5,
        }

        self.TRIGGER_EVENT_ID = {str(el): i + 1 for i, el in enumerate([276, 277, 768, 769, 770, 781, 783, 1023,
                                                                        1077, 1078, 1079, 1081, 32766])}

        self.DROP_SUBJECTS = []


class BciCompIV1:
    """
    In this dataset there are only 2 classes out of left, right, foot at each subject.
    It is not suggested to train it in a cross subject fashion...
    """

    def __init__(self, config_ver=-1):
        self.DIR = "BCI_comp/4/1"
        self.CONFIG_VER = 1.1 if config_ver == -1. else config_ver

        self.FILE_PATH = 'S{subj}/S{subj}{rec}_raw.fif'
        self.SUBJECT_NUM = 7
        self.SUBJECT_EXP = {subj: (subj,) for subj in range(1, self.SUBJECT_NUM + 1)}

        self.TRIGGER_TASK_CONVERTER = {  # imagined
            'class1': 1,
            'class2': 2,
        }

        self.TRIGGER_EVENT_ID = 'auto'

        self.DROP_SUBJECTS = [3, 4, 5]  # artificial data


class Giga:

    def __init__(self, config_ver=-1):
        self.DIR = "giga"
        self.CONFIG_VER = 1.1 if config_ver == -1. else config_ver

        self.FILE_PATH = 'S{subj}/S{subj}R{rec}_raw.fif'
        self.SUBJECT_NUM = 2 * 54
        self.SUBJECT_EXP = {s + 1: [s * 2 + 1, (s + 1) * 2] for s in range(54)}  # must be sorted!

        self.TRIGGER_TASK_CONVERTER = {  # imagined
            RIGHT_HAND: 1,
            LEFT_HAND: 2,
        }

        self.TRIGGER_EVENT_ID = {el: i + 1 for i, el in enumerate(['right', 'left'])}

        self.DROP_SUBJECTS = []

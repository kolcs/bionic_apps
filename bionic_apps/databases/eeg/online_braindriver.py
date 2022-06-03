from .defaults import LEFT_HAND, RIGHT_HAND, BOTH_HANDS, BOTH_LEGS, REST, CALM, ACTIVE, \
    LEFT_LEG, RIGHT_LEG
from ...games.braindriver.control import ControlCommand

DIR_FEATURE_DB = 'tmp/'


class GameDB:

    def __init__(self, config_ver=-1):
        self.DIR = "Game/mixed"
        self.FILE_PATH = 'subject{subj}/rec{rec}.vhdr'
        self.CONFIG_VER = 0 if config_ver == -1. else config_ver

        self.TRIGGER_TASK_CONVERTER = {  # imagined
            REST: 1,
            RIGHT_HAND: 5,
            LEFT_HAND: 7,
            # RIGHT_LEG: 9,
            # LEFT_LEG: 11
            # BOTH_HANDS: 9,
            BOTH_LEGS: 11
        }

        self.COMMAND_CONV = {
            REST: ControlCommand.STRAIGHT,
            RIGHT_HAND: ControlCommand.RIGHT,
            LEFT_HAND: ControlCommand.LEFT,
            BOTH_LEGS: ControlCommand.HEADLIGHT
        }

        self.TRIGGER_EVENT_ID = {'Stimulus/S ' + (2 - len(str(i + 1))) * ' ' + str(i + 1): i + 1 for i in range(16)}

        self.DROP_SUBJECTS = []


class Game_ParadigmC:

    def __init__(self, config_ver=-1):
        self.DIR = "Game/paradigmC/"
        self.CONFIG_VER = 1.1 if config_ver == -1. else config_ver

        if self.CONFIG_VER > 1:
            self.FILE_PATH = 'S{subj}/S{subj}R{rec}_raw.fif'
            self.SUBJECT_EXP = {  # must be sorted!
                1: list(range(1, 6)),
            }
        else:
            self.FILE_PATH = 'subject{subj}/rec{rec}.vhdr'

        self.TRIGGER_TASK_CONVERTER = {  # imagined
            # REST: 1,
            # EYE_OPEN: 2,
            # EYE_CLOSED: 3,
            RIGHT_HAND: 5,
            LEFT_HAND: 7,
            CALM: 9,
            BOTH_LEGS: 11
        }

        self.COMMAND_CONV = {
            CALM: ControlCommand.STRAIGHT,
            RIGHT_HAND: ControlCommand.RIGHT,
            LEFT_HAND: ControlCommand.LEFT,
            BOTH_LEGS: ControlCommand.HEADLIGHT
        }

        self.TRIGGER_EVENT_ID = {'Stimulus/S ' + (2 - len(str(i + 1))) * ' ' + str(i + 1): i + 1 for i in range(16)}

        self.DROP_SUBJECTS = []


class ParadigmC:

    def __init__(self, config_ver=-1):
        self.DIR = "ParC/"
        self.FILE_PATH = 'subject{subj}/rec{rec}.vhdr'
        self.CONFIG_VER = 1.1 if config_ver == -1. else config_ver

        self.TRIGGER_TASK_CONVERTER = {  # imagined
            # REST: 1,
            # EYE_OPEN: 2,
            # EYE_CLOSED: 3,
            RIGHT_HAND: 5,
            LEFT_HAND: 7,
            CALM: 9,
            BOTH_LEGS: 11
        }

        self.COMMAND_CONV = {
            CALM: ControlCommand.STRAIGHT,
            RIGHT_HAND: ControlCommand.RIGHT,
            LEFT_HAND: ControlCommand.LEFT,
            BOTH_LEGS: ControlCommand.HEADLIGHT
        }

        self.TRIGGER_EVENT_ID = {'Stimulus/S ' + (2 - len(str(i + 1))) * ' ' + str(i + 1): i + 1 for i in range(16)}

        self.DROP_SUBJECTS = [1]


class Game_ParadigmD:

    def __init__(self, config_ver=-1):
        self.DIR = "Game/paradigmD/"
        self.CONFIG_VER = 1.1 if config_ver == -1. else config_ver

        if self.CONFIG_VER > 1:
            self.FILE_PATH = 'S{subj}/S{subj}R{rec}_raw.fif'
            self.SUBJECT_EXP = {  # must be sorted!
                1: [1, 2, 5, 6, 7, 8, 10, 11, 13, 15],
                2: [3, 4, 9, 12, 14, 16],
            }
        else:
            self.FILE_PATH = 'subject{subj}/rec{rec}.vhdr'

        self.TRIGGER_TASK_CONVERTER = {  # imagined
            # REST: 1,
            # EYE_OPEN: 2,
            # EYE_CLOSED: 3,
            ACTIVE + '1': 5,
            ACTIVE + '2': 9,
            CALM + '1': 7,
            CALM + '2': 11
        }

        self.TRIGGER_EVENT_ID = {'Stimulus/S ' + (2 - len(str(i + 1))) * ' ' + str(i + 1): i + 1 for i in range(16)}

        self.DROP_SUBJECTS = []


class PilotDB_ParadigmA:

    def __init__(self, config_ver=-1):
        self.DIR = "Cybathlon_pilot/paradigmA/"
        self.CONFIG_VER = 1.1 if config_ver == -1. else config_ver

        if self.CONFIG_VER > 1:
            self.FILE_PATH = 'S{subj}/S{subj}R{rec}_raw.fif'
            self.SUBJECT_EXP = {  # must be sorted!
                1: [1, 4],
                2: [2, 3],
            }
        else:
            self.FILE_PATH = 'pilot{subj}/rec{rec}.vhdr'

        self.TRIGGER_TASK_CONVERTER = {  # imagined
            # REST: 1,
            # EYE_OPEN: 2,
            # EYE_CLOSED: 3,
            RIGHT_HAND: 5,
            LEFT_HAND: 7,
            RIGHT_LEG: 9,
            LEFT_LEG: 11
        }

        self.TRIGGER_EVENT_ID = {'Stimulus/S ' + (2 - len(str(i + 1))) * ' ' + str(i + 1): i + 1 for i in range(16)}

        self.DROP_SUBJECTS = []


class PilotDB_ParadigmB:

    def __init__(self, config_ver=-1):
        self.DIR = "Cybathlon_pilot/paradigmB/"
        self.CONFIG_VER = 1.1 if config_ver == -1. else config_ver

        if self.CONFIG_VER > 1:
            self.FILE_PATH = 'S{subj}/S{subj}R{rec}_raw.fif'
            self.SUBJECT_EXP = {  # must be sorted!
                1: [1, 3],
                2: [2, 4],
            }
        else:
            self.FILE_PATH = 'pilot{subj}/rec{rec}.vhdr'

        self.TRIGGER_TASK_CONVERTER = {  # imagined
            # REST: 1,
            # EYE_OPEN: 2,
            # EYE_CLOSED: 3,
            RIGHT_HAND: 5,
            LEFT_HAND: 7,
            BOTH_HANDS: 9,
            BOTH_LEGS: 11
        }

        self.TRIGGER_EVENT_ID = {'Stimulus/S ' + (2 - len(str(i + 1))) * ' ' + str(i + 1): i + 1 for i in range(16)}

        self.DROP_SUBJECTS = []


class TTK_DB:

    def __init__(self, config_ver=-1):
        self.DIR = "TTK/"
        self.CONFIG_VER = 1.1 if config_ver == -1. else config_ver

        if self.CONFIG_VER >= 1:
            self.SUBJECT_EXP = {  # must be sorted!
                1: [1, 10, 19],
                2: [2],
                3: [3, 11, 23],
                4: [4, 6, 8],
                5: [5, 7, 17, 21],
                6: [9, 12, 13, 14, 20],
                7: [15, 16, 18, 22],
                8: [24],
                9: [25],
            }
            self.FILE_PATH = 'S{subj}/S{subj}R{rec}_raw.fif'
            self.DROP_SUBJECTS = []
        else:
            self.FILE_PATH = 'subject{subj}/rec{rec}.vhdr'
            self.DROP_SUBJECTS = [1, 9, 17]

        self.TRIGGER_TASK_CONVERTER = {  # imagined
            # REST: 1,
            # EYE_OPEN: 2,
            # EYE_CLOSED: 3,
            RIGHT_HAND: 5,
            LEFT_HAND: 7,
            RIGHT_LEG: 9,
            LEFT_LEG: 11
        }

        self.TRIGGER_EVENT_ID = {'Stimulus/S ' + (2 - len(str(i + 1))) * ' ' + str(i + 1): i + 1 for i in range(16)}


class EmotivParC:

    def __init__(self, config_ver=-1):
        self.DIR = "bionic_apps/external_connections/emotiv/paradigmC/"
        # self.CONFIG_VER = 0 if config_ver == -1. else config_ver

        self.FILE_PATH = 'sub-P{subj}_run-{rec}_eeg.xdf'

        self.TRIGGER_TASK_CONVERTER = {  # imagined
            # REST: 1,
            # EYE_OPEN: 2,
            # EYE_CLOSED: 3,
            RIGHT_HAND: 5,
            LEFT_HAND: 7,
            CALM: 9,
            BOTH_LEGS: 11
        }

        self.TRIGGER_EVENT_ID = {'S ' + (2 - len(str(i + 1))) * ' ' + str(i + 1): i + 1 for i in range(16)}

        self.DROP_SUBJECTS = [1, 2, 3]

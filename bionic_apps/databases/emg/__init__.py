class PutEMG:

    def __init__(self, config_ver=-1):
        self.DIR = "putemg"
        self.CONFIG_VER = 1.1 if config_ver == -1. else config_ver

        self.FILE_PATH = 'subject{subj}_raw.fif'
        self.SUBJECT_NUM = 87
        self.DROP_SUBJECTS = [43, 59, 63, 80]

class MindRoveCoreg:

    def __init__(self, config_ver=-1):
        self.DIR = "MindRove-coreg"
        self.CONFIG_VER = 1.1 if config_ver == -1. else config_ver

        self.FILE_PATH = 'subject{subj}_raw.fif'
        self.SUBJECT_NUM = 6
        self.DROP_SUBJECTS = []

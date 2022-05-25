from enum import Enum

from eeg.offline import *
from eeg.online_braindriver import *


# db selection options
class EEG_Databases(Enum):
    PHYSIONET = 'physionet'
    PILOT_PAR_A = 'pilot_par_a'
    PILOT_PAR_B = 'pilot_par_b'
    TTK = 'ttk'
    GAME = 'game'
    GAME_PAR_C = 'game_par_c'
    GAME_PAR_D = 'game_par_d'
    BCI_COMP_IV_1 = 'BCICompIV1'
    BCI_COMP_IV_2A = 'BCICompIV2a'
    BCI_COMP_IV_2B = 'BCICompIV2b'
    ParadigmC = 'par_c'
    EMOTIV_PAR_C = 'emotiv_par_c'
    GIGA = 'giga'


def get_eeg_db_name_by_filename(filename):
    if Game_ParadigmC().DIR in filename:
        db_name = EEG_Databases.GAME_PAR_C
    elif Game_ParadigmD().DIR in filename:
        db_name = EEG_Databases.GAME_PAR_D
    elif PilotDB_ParadigmA().DIR in filename:
        db_name = EEG_Databases.PILOT_PAR_A
    elif PilotDB_ParadigmB().DIR in filename:
        db_name = EEG_Databases.PILOT_PAR_B
    elif Physionet().DIR in filename:
        db_name = EEG_Databases.PHYSIONET
    elif ParadigmC().DIR in filename:
        db_name = EEG_Databases.ParadigmC
    elif BciCompIV1().DIR in filename:
        db_name = EEG_Databases.BCI_COMP_IV_1
    elif BciCompIV2a().DIR in filename:
        db_name = EEG_Databases.BCI_COMP_IV_2A
    elif BciCompIV2b().DIR in filename:
        db_name = EEG_Databases.BCI_COMP_IV_2B
    elif TTK_DB().DIR in filename:
        db_name = EEG_Databases.TTK
    elif Giga().DIR in filename:
        db_name = EEG_Databases.GIGA
    else:
        raise ValueError('No database defined with path {}'.format(filename))
    return db_name

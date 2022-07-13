from enum import Enum
from pathlib import Path

from .coreg_mindrove import MindRoveCoreg
from .eeg.offline import *
from .eeg.online_braindriver import *


# db selection options
class Databases(Enum):
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

    MINDROVE_COREG = 'mindrove'


def get_eeg_db_name_by_filename(filename):
    filename = Path(filename).as_posix()
    if Game_ParadigmC().DIR in filename:
        db_name = Databases.GAME_PAR_C
    elif Game_ParadigmD().DIR in filename:
        db_name = Databases.GAME_PAR_D
    elif PilotDB_ParadigmA().DIR in filename:
        db_name = Databases.PILOT_PAR_A
    elif PilotDB_ParadigmB().DIR in filename:
        db_name = Databases.PILOT_PAR_B
    elif Physionet().DIR in filename:
        db_name = Databases.PHYSIONET
    elif ParadigmC().DIR in filename:
        db_name = Databases.ParadigmC
    elif BciCompIV1().DIR in filename:
        db_name = Databases.BCI_COMP_IV_1
    elif BciCompIV2a().DIR in filename:
        db_name = Databases.BCI_COMP_IV_2A
    elif BciCompIV2b().DIR in filename:
        db_name = Databases.BCI_COMP_IV_2B
    elif TTK_DB().DIR in filename:
        db_name = Databases.TTK
    elif Giga().DIR in filename:
        db_name = Databases.GIGA
    elif MindRoveCoreg().DIR in filename:
        db_name = Databases.MINDROVE_COREG
    else:
        raise ValueError('No database defined with path {}'.format(filename))
    return db_name

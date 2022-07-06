from pathlib import Path
from shutil import rmtree

from ..databases import EEG_Databases, get_eeg_db_name_by_filename
from ..utils import init_base_config

EXCLUDE_DB_LIST = [EEG_Databases.ParadigmC, EEG_Databases.EMOTIV_PAR_C]


def get_available_databases():
    base_dir = Path(init_base_config())
    avail_dbs = set()
    for file in base_dir.rglob('*'):
        try:
            avail_dbs.add(get_eeg_db_name_by_filename(file.as_posix()))
        except ValueError:
            pass
    avail_dbs = [db_name for db_name in avail_dbs if db_name not in EXCLUDE_DB_LIST]
    return avail_dbs


AVAILABLE_DBS = get_available_databases()


def cleanup_fastload_data(path='tmp/'):
    path = Path(init_base_config()).joinpath(path)
    if path.exists():
        print('Removing old files. It may take for a while...')
        rmtree(str(path))

from pathlib import Path
from shutil import rmtree

from config import DIR_FEATURE_DB
from preprocess import Databases, init_base_config, get_db_name_by_filename

EXCLUDE_DB_LIST = [Databases.BCI_COMP_IV_1, Databases.ParadigmC, Databases.EMOTIV_PAR_C]


def get_available_databases():
    base_dir = Path(init_base_config('..'))
    avail_dbs = set()
    for file in base_dir.rglob('*'):
        try:
            avail_dbs.add(get_db_name_by_filename(file.as_posix()))
        except ValueError:
            pass
    avail_dbs = [db_name for db_name in avail_dbs if db_name not in EXCLUDE_DB_LIST]
    return avail_dbs


AVAILABLE_DBS = get_available_databases()


def cleanup_fastload_data():
    path = Path(init_base_config('..')).joinpath(DIR_FEATURE_DB)
    if path.exists():
        print('Removing old files. It may take for a while...')
        rmtree(str(path))

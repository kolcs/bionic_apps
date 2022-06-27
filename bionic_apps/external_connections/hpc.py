import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from numpy.random import randint

from bionic_apps import config
from bionic_apps.utils import load_from_json

CHECKPOINT = 'checkpoint.json'
FEATURE_DIR = 'feature_dir'
SUBJECT_NUM = 'subj_num'
cp_info = dict()


def _cleanup_files_older_than(path, days=7):
    for (root, dirs, files) in os.walk(str(path)):
        for f in files:
            file = os.path.join(root, f)
            if datetime.now() - datetime.fromtimestamp(os.path.getmtime(file)) > timedelta(days=days):
                os.remove(file)


def _gen_save_path():
    user = subprocess.check_output('whoami').decode('utf-8').strip('\n')
    path = Path(f'/scratch{randint(1, 5)}/{user}/bci/{datetime.now().strftime("%Y%m%d-%H%M%S-%f")}')
    _cleanup_files_older_than(path.parent)
    return path


# def run_with_checkpoint(test_func, checkpoint=None, verbose=False, **test_kwargs):
#     global cp_info, CHECKPOINT
#     if type(checkpoint) is str:
#         CHECKPOINT = checkpoint
#
#     fast_load = True
#     try:
#         cp_info = load_from_json(CHECKPOINT)
#     except FileNotFoundError:
#         path = _gen_save_path()
#         # rmtree(path, ignore_errors=True)
#         cp_info[FEATURE_DIR] = path
#         cp_info[SUBJECT_NUM] = 1
#         fast_load = False
#     config.SAVE_PATH = cp_info[FEATURE_DIR]
#
#     # running the test with checkpoints...
#     test_func(subject_from=cp_info[SUBJECT_NUM], fast_load=fast_load,
#               verbose=verbose, **test_kwargs)
#
#     os.remove(CHECKPOINT)


def run_without_checkpoint(test_func, **test_kwargs):
    config.SAVE_PATH = _gen_save_path()
    test_func(**test_kwargs)

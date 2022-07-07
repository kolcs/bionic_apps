import importlib
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from numpy.random import randint

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


def run_without_checkpoint(test_func):
    def wrap(*args, **kwargs):
        assert 'db_file' in kwargs, f'db_file param is not in kwargs of {test_func.__name__}()'
        kwargs['db_file'] = _gen_save_path().joinpath('database.h5')
        test_func(*args, **kwargs)

    return wrap


def make_one_test():
    _, par_module, package, param_ind = sys.argv

    par_module = importlib.import_module(par_module, package)

    param_ind = int(param_ind)
    test_func = run_without_checkpoint(par_module.test_func)
    hpc_kwargs = par_module.default_kwargs
    hpc_kwargs.update(par_module.test_kwargs[param_ind])

    folder = Path(par_module.LOG_DIR).joinpath(hpc_kwargs['db_name'].value, par_module.TEST_NAME)
    folder.mkdir(parents=True, exist_ok=True)
    log_file = str(folder.joinpath('{}-{}.csv'.format(param_ind, datetime.now().strftime("%Y%m%d-%H%M%S"))))
    hpc_kwargs['log_file'] = log_file

    test_func(**hpc_kwargs)


# stuff to main script:
# python -c "from bionic_apps.external_connections.hpc.utils import start_test; start_test()"

def start_test(par_module='example_params',
               package='bionic_apps.external_connections.hpc'):  # call this from script of from python
    par_module = importlib.import_module(par_module, package)
    job_list = 'Submitted batch jobs:\n'
    Path(par_module.LOG_DIR).joinpath('std', 'out').mkdir(parents=True, exist_ok=True)  # sdt out and error files
    Path(par_module.LOG_DIR).joinpath('std', 'err').mkdir(parents=True, exist_ok=True)  # sdt out and error files

    user_ans = input(f'{len(par_module.test_kwargs)} '
                     f'HPC jobs will be created. Do you want to continue [y] / n?  ')
    if user_ans in ['', 'y']:
        pass
    else:
        if user_ans != 'n':
            print('Incorrect command.')
        print(f'{__file__} is terminated.')
        exit(0)

    for i in range(len(par_module.test_params)):
        cmd = f'sbatch {par_module.hpc_submit_script}'
        cmd += f' {__file__} {par_module} {package} {i}'
        ans = subprocess.check_output(cmd, shell=True)
        job_list += ans.decode('utf-8').strip('\n').strip('\r').strip('Submitted batch job') + ' '
    print(job_list)
    job_file = Path(par_module.LOG_DIR).joinpath('hpc_jobs.txt')
    job_file.write_text(job_list)


if __name__ == '__main__':
    make_one_test()

import fileinput
import importlib
import inspect
import os
import shutil
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
    walk = list(os.walk(str(path)))
    for (root, dirs, files) in walk[::-1]:
        all_files_deleted = True
        for f in files:
            file = os.path.join(root, f)
            if datetime.now() - datetime.fromtimestamp(os.path.getmtime(file)) > timedelta(days=days):
                os.remove(file)
            else:
                all_files_deleted = False
        if all_files_deleted:
            try:
                os.rmdir(root)
            except OSError:
                pass


def _gen_save_path(test_name, ind):
    user = subprocess.check_output('whoami').decode('utf-8').strip('\n')
    path = Path(f'/scratch{randint(1, 5)}').joinpath(user, 'bci', test_name, str(ind))
    _cleanup_files_older_than(path.parent.parent)
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


# def run_without_checkpoint(test_func):
#     def wrap(*args, **kwargs):
#         assert 'db_file' in list(inspect.signature(test_func).parameters), \
#             f'db_file param is not in kwargs of {test_func.__name__}()'
#         save_path = _gen_save_path()
#         kwargs['db_file'] = save_path.joinpath('database.h5')
#         kwargs['classifier_kwargs']['save_path'] = save_path
#         test_func(*args, **kwargs)
#
#     return wrap

def run_without_checkpoint(test_func, test_name, ind, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    assert 'db_file' in list(inspect.signature(test_func).parameters), \
        f'db_file param is not in kwargs of {test_func.__name__}()'
    save_path = _gen_save_path(test_name, ind)
    kwargs['db_file'] = save_path.joinpath('database.h5')
    kwargs['classifier_kwargs']['save_path'] = save_path
    test_func(*args, **kwargs)


def make_one_test():
    _, module, package, param_ind = sys.argv

    par_module = importlib.import_module(module, package)
    # par_module = lazy_import(par_module, package)

    param_ind = int(param_ind)
    hpc_kwargs = par_module.default_kwargs
    hpc_kwargs.update(par_module.test_kwargs[param_ind])

    folder = Path(par_module.LOG_DIR).joinpath(hpc_kwargs['db_name'].value, par_module.TEST_NAME)
    folder.mkdir(parents=True, exist_ok=True)
    log_file = str(folder.joinpath('{}-{}.csv'.format(param_ind, datetime.now().strftime("%Y%m%d-%H%M%S"))))
    hpc_kwargs['log_file'] = log_file

    run_without_checkpoint(par_module.test_func, par_module.TEST_NAME, param_ind, kwargs=hpc_kwargs)


# stuff to main script:
# python -c "from bionic_apps.external_connections.hpc.utils import start_test; start_test()"

def start_test(module='.example_params',
               package='bionic_apps.external_connections.hpc'):  # call this from script of from python
    # par_module = lazy_import(par_module, package)
    par_module = importlib.import_module(module, package)
    job_list = 'Submitted batch jobs:\n'

    std_out = Path(par_module.LOG_DIR).joinpath('std', 'out')
    std_err = Path(par_module.LOG_DIR).joinpath('std', 'err')
    std_out.mkdir(parents=True, exist_ok=True)  # sdt out and error files
    std_err.mkdir(parents=True, exist_ok=True)  # sdt out and error files

    user_ans = input(f'{len(par_module.test_kwargs)} '
                     f'HPC jobs will be created. Do you want to continue [y] / n?  ')
    if user_ans in ['', 'y']:
        pass
    else:
        if user_ans != 'n':
            print('Incorrect command.')
        print(f'{__file__} is terminated.')
        exit(0)

    submit_script = Path(par_module.LOG_DIR).joinpath(par_module.hpc_submit_script)
    shutil.copy(par_module.hpc_submit_script, submit_script)
    with fileinput.FileInput(submit_script, inplace=True) as f:
        for line in f:
            if "#SBATCH -o outfile-%j" in line:
                print(f"#SBATCH -o {std_out}/outfile-%j", end='\n')
            elif "#SBATCH -e errfile-%j" in line:
                print(f"#SBATCH -e {std_err}/errfile-%j", end='\n')
            else:
                print(line, end='')

    for i in range(len(par_module.test_kwargs)):
        cmd = f'sbatch {submit_script}'
        cmd += f' {__file__} {module} {package} {i}'
        ans = subprocess.check_output(cmd, shell=True)
        job_list += ans.decode('utf-8').strip('\n').strip('\r').strip('Submitted batch job') + ' '
    print(job_list)
    job_file = Path(par_module.LOG_DIR).joinpath('hpc_jobs.txt')
    job_file.write_text(job_list)


if __name__ == '__main__':
    make_one_test()

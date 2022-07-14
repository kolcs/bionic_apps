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

from bionic_apps.preprocess.io import DataLoader
from bionic_apps.utils import load_from_json, save_to_json

PROCESSED_SUBJ = 'subj'


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


def _gen_hpc_save_path(log_path):
    user = subprocess.check_output('whoami').decode('utf-8').strip('\n')
    base_path = Path(f'/scratch{randint(1, 5)}').joinpath(user)
    _cleanup_files_older_than(base_path)
    return base_path.joinpath(log_path)


def run_with_checkpoint(test_func, log_path, subjects, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    cp_info = dict(
        filename=str(log_path.joinpath('check_point.json'))
    )

    try:
        cp_info = load_from_json(cp_info['filename'])
        subjects = [s for s in subjects if s > cp_info[PROCESSED_SUBJ]]
    except FileNotFoundError:
        cp_info[PROCESSED_SUBJ] = 0
        subjects = 'all'

    assert 'db_file' in list(inspect.signature(test_func).parameters), \
        f'db_file param is not in kwargs of {test_func.__name__}()'

    save_path = _gen_hpc_save_path(log_path)
    cp_info['save_path'] = str(save_path)
    save_to_json(cp_info['filename'], cp_info)

    kwargs['db_file'] = save_path.joinpath('database.h5')
    kwargs['classifier_kwargs']['save_path'] = save_path
    kwargs['classifier_kwargs']['verbose'] = 2
    kwargs['subjects'] = subjects
    kwargs['hpc_check_point'] = cp_info

    try:
        test_func(*args, **kwargs)
        os.remove(cp_info['filename'])
        shutil.rmtree(str(save_path))
    except Exception as e:
        shutil.rmtree(str(save_path))
        raise e


def run_without_checkpoint(test_func, log_path, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    assert 'db_file' in list(inspect.signature(test_func).parameters), \
        f'db_file param is not in kwargs of {test_func.__name__}()'
    save_path = _gen_hpc_save_path(log_path)
    kwargs['db_file'] = save_path.joinpath('database.h5')
    kwargs['classifier_kwargs']['save_path'] = save_path
    kwargs['classifier_kwargs']['verbose'] = 2

    try:
        test_func(*args, **kwargs)
        shutil.rmtree(str(save_path))
    except Exception as e:
        shutil.rmtree(str(save_path))
        raise e


def make_one_test():
    _, module, package, param_ind = sys.argv

    print(f'Starting job with param_ind: {param_ind}')
    par_module = importlib.import_module(module, package)
    # par_module = lazy_import(par_module, package)

    hpc_kwargs = par_module.default_kwargs
    hpc_kwargs.update(par_module.test_kwargs[int(param_ind)])

    db_name = hpc_kwargs['db_name']
    log_path = Path(par_module.LOG_DIR).joinpath(db_name.value, par_module.TEST_NAME, param_ind)
    log_path.mkdir(parents=True, exist_ok=True)
    hpc_kwargs['log_file'] = str(log_path.joinpath(f'{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv'))

    subjects = DataLoader().use_db(db_name).get_subject_list()
    # run_without_checkpoint(par_module.test_func, log_path, kwargs=hpc_kwargs)
    run_with_checkpoint(par_module.test_func, log_path,
                        subjects=subjects, kwargs=hpc_kwargs)


# stuff to main script:
# python -c "from bionic_apps.external_connections.hpc.utils import start_test; start_test()"

def start_test(module='example_params',
               package='bionic_apps.external_connections.hpc'):
    if len(package) > 0 and module[0] != '.':
        module = '.' + module
    par_module = importlib.import_module(module, package)

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

    job_list = 'Submitted batch jobs:\n'
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

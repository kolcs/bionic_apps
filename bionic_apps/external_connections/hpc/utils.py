import fileinput
import importlib
import inspect
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from getpass import getpass
from pathlib import Path
from time import sleep

import numpy as np
import pexpect

from bionic_apps.preprocess.io import DataLoader
from bionic_apps.utils import load_from_json, save_to_json

PROCESSED_SUBJ = 'subj'
JOB_INFO = 'Submitted batch jobs:\n'

GPU_TYPES = {
    1: 'gpu:v100:1',
    2: 'gpu:v100:1',
    3: 'gpu:a100:1',
}


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


def _gen_hpc_save_path(log_path, tried_scratches=()):
    user = subprocess.check_output('whoami').decode('utf-8').strip('\n')
    scratch_list = np.arange(1, 5)
    np.random.shuffle(scratch_list)
    base_path = Path()
    scratch = 0
    for scratch in scratch_list:
        if scratch in tried_scratches:
            continue
        base_path = Path(f'/scratch{scratch}')
        if base_path.exists():
            base_path = base_path.joinpath(user)
            try:
                base_path.mkdir(exist_ok=True)
                break
            except PermissionError:
                pass
        tried_scratches += (scratch,)
    if len(tried_scratches) == 4:
        raise EnvironmentError('Out of resource.')
    _cleanup_files_older_than(base_path)
    return base_path.joinpath(log_path), scratch, tried_scratches


def _check_param_in_func(par_name, func):
    pars = list(inspect.signature(func).parameters)
    assert par_name in pars, \
        f'{par_name} param is not in kwargs of {func.__name__}(). ' \
        f'kwargs are: {pars}'


def ssh_and_cleanup(node=None, scratch=None):
    if isinstance(node, str):
        node = [node]
    if node is None:
        node = ['erdos', 'neumann', 'renyi', 'wald', 'lanczos']
    elif not isinstance(node, (tuple, list)):
        raise TypeError('node param can be str, tuple or list')

    if isinstance(scratch, (int, str)):
        scratch = [scratch]
    if scratch is None:
        scratch = ['/scratch1', '/scratch2', '/scratch3', '/scratch4']
    elif not isinstance(scratch, (tuple, list)):
        raise TypeError('node param can be str, tuple or list')
    scratch = [f'/scratch{s}' if isinstance(s, int) else s for s in scratch]

    user = subprocess.check_output('whoami').decode('utf-8').strip('\n')
    pwd = getpass('HPC password: ')

    for n in node:
        ssh = pexpect.spawn(f'ssh {user}@{n}')
        ssh.expect('Password:')
        ssh.sendline(pwd)
        i = ssh.expect(['[Pp]assword:', 'Permission denied', '[#\$] '])
        while i == 0:
            pwd = getpass('HPC password: ')
            ssh.sendline(pwd)
            i = ssh.expect(['[Pp]assword:', 'Permission denied', '[#\$] '])
        if i == 1:
            print('Permission denied on host. Can\'t login')
            ssh.kill(0)
            exit(12)
        elif i == 2:
            print(f'Login to {n}')

        for s in scratch:
            ssh.sendline(f'rm -r {s}/{user}')
            ssh.expect('[#\$] ')

        ssh.sendline('logout')
        print('Cleanup finished. Logging out.')


def run_with_checkpoint(test_func, log_path, subjects, tried_scratches=(), args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    if 'classifier_kwargs' not in kwargs:
        kwargs['classifier_kwargs'] = {}

    cp_info = dict(
        filename=str(log_path.joinpath('check_point.json'))
    )

    try:
        cp_info = load_from_json(cp_info['filename'])
        subjects = [s for s in subjects if s > cp_info[PROCESSED_SUBJ]]
    except FileNotFoundError:
        cp_info[PROCESSED_SUBJ] = 0
        subjects = 'all'

    _check_param_in_func('db_file', test_func)
    _check_param_in_func('classifier_kwargs', test_func)
    _check_param_in_func('subjects', test_func)
    _check_param_in_func('hpc_check_point', test_func)

    save_path, scratch, tried_scratches = _gen_hpc_save_path(log_path, tried_scratches)
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
    except OSError:
        if len(tried_scratches) < 3:
            shutil.rmtree(save_path)
            run_with_checkpoint(test_func, log_path, subjects,
                                tried_scratches + (scratch,),
                                args, kwargs)
        else:
            raise MemoryError('All SSDs are out of space.')
    except Exception as e:
        raise e
    finally:
        if save_path.exists():
            shutil.rmtree(save_path)


# def run_without_checkpoint(test_func, log_path, args=(), kwargs=None):
#     if kwargs is None:
#         kwargs = {}
#     assert 'db_file' in list(inspect.signature(test_func).parameters), \
#         f'db_file param is not in kwargs of {test_func.__name__}()'
#     save_path = _gen_hpc_save_path(log_path)
#     kwargs['db_file'] = save_path.joinpath('database.h5')
#     kwargs['classifier_kwargs']['save_path'] = save_path
#     kwargs['classifier_kwargs']['verbose'] = 2
#
#     try:
#         test_func(*args, **kwargs)
#     except Exception as e:
#         raise e
#     finally:
#         shutil.rmtree(str(save_path))


def make_one_test():
    _, module, package, param_ind = sys.argv

    print(f'Starting job with param_ind: {param_ind}')
    par_module = importlib.import_module(module, package)
    # par_module = lazy_import(par_module, package)

    hpc_kwargs = par_module.default_kwargs
    hpc_kwargs.update(par_module.test_kwargs[int(param_ind)])

    db_name = hpc_kwargs['db_name']
    log_path = Path(par_module.LOG_DIR).joinpath(par_module.TEST_NAME, db_name.value, param_ind)
    log_path.mkdir(parents=True, exist_ok=True)
    hpc_kwargs['log_file'] = str(log_path.joinpath(f'{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv'))

    subjects = DataLoader().use_db(db_name).get_subject_list()

    run_with_checkpoint(par_module.test_func, log_path,
                        subjects=subjects, kwargs=hpc_kwargs)


# stuff to main script:
# python -c "from bionic_apps.external_connections.hpc.utils import start_test; start_test()"

def start_test(module='example_params',
               package='bionic_apps.external_connections.hpc',
               submit_only_unifinished=False):
    if len(package) > 0 and module[0] != '.':
        module = '.' + module
    par_module = importlib.import_module(module, package)

    if par_module.partition.startswith('gpu'):
        if par_module.gpu_type == 2 and par_module.partition in ['gpu_long', 'gpu_lowpriority']:
            raise ValueError('GPU 2 partition is only available with timeout setup.'
                             'Please set ``partition = \'gpu\'``')
        elif par_module.gpu_type == 3 and par_module.partition == 'gpu_long':
            raise ValueError('GPU 3 partition can not be used with ``partition '
                             '= \'gpu_long\'``. Use \'gpu\' or \'gpu_lowpriority\' instead.')

    std_out = Path(par_module.LOG_DIR).joinpath('std', 'out')
    std_err = Path(par_module.LOG_DIR).joinpath('std', 'err')
    std_out.mkdir(parents=True, exist_ok=True)  # sdt out and error files
    std_err.mkdir(parents=True, exist_ok=True)  # sdt out and error files

    if submit_only_unifinished:
        log_path = Path(par_module.LOG_DIR).joinpath(par_module.TEST_NAME)
        job_file = log_path.joinpath('hpc_jobs.txt')
        if job_file.exists():
            jobs = job_file.read_text().strip(JOB_INFO)
            subprocess.check_output(f'scancel {jobs}', shell=True)
        cp_files = list(log_path.rglob('check_point.json'))
        cp_inds = [int(file.parent.name) for file in cp_files]
        n_hpc_jobs = len(cp_files)
        assert n_hpc_jobs > 0, f'No checkpoint files were found on path {log_path}'
    else:
        n_hpc_jobs = len(par_module.test_kwargs)
        cp_inds = []

    user_ans = input(f'{n_hpc_jobs} '
                     f'HPC jobs will be created. Do you want to continue [y] / n?  ')
    if user_ans in ['', 'y']:
        pass
    else:
        if user_ans != 'n':
            print('Incorrect command.')
        print(f'{__file__} is terminated.')
        exit(0)

    partition = par_module.partition[:3]
    if partition == 'gpu':
        partition += str(par_module.gpu_type)
    submit_script = Path(par_module.LOG_DIR).joinpath(f'{partition}_submit.sh')
    shutil.copy(f'base_{par_module.partition[:3]}.sh', submit_script)

    with fileinput.FileInput(submit_script, inplace=True) as f:
        for line in f:
            if line.startswith('#SBATCH -p'):
                print(f'#SBATCH -p {par_module.partition}', end='\n')
            elif line.startswith('#SBATCH --gres='):
                print(f'#SBATCH --gres={GPU_TYPES[par_module.gpu_type]}', end='\n')
                if par_module.gpu_type == 2:
                    print(f'#SBATCH --nodelist=neumann', end='\n')
            elif line.startswith('#SBATCH -c'):
                print(f'#SBATCH -c {par_module.cpu_cores}', end='\n')
            elif line.startswith("#SBATCH -o"):
                print(f"#SBATCH -o {std_out}/outfile-%j", end='\n')
            elif line.startswith("#SBATCH -e"):
                print(f"#SBATCH -e {std_err}/errfile-%j", end='\n')
            else:
                print(line, end='')

    job_list = JOB_INFO
    for i in range(len(par_module.test_kwargs)):
        if submit_only_unifinished and i not in cp_inds:
            continue
        cmd = f'sbatch {submit_script}'
        cmd += f' {__file__} {module} {package} {i}'
        ans = subprocess.check_output(cmd, shell=True)
        sleep(.2)
        job_list += ans.decode('utf-8').strip('\n').strip('\r').strip('Submitted batch job') + ' '
    print(job_list)
    job_file = Path(par_module.LOG_DIR).joinpath(par_module.TEST_NAME, 'hpc_jobs.txt')
    job_file.parent.mkdir(exist_ok=True)
    job_file.write_text(job_list)


if __name__ == '__main__':
    make_one_test()

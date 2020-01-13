import logging
import datetime
import socket
from struct import unpack

from preprocess import make_dir

UDP_IP = '127.0.0.1'
UDP_PORT = 8053
BUFFER_SIZE = 36


class GameLogger(object):

    def __init__(self, bv_rcc=None, player=1):
        self._soc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._soc.bind((UDP_IP, UDP_PORT))
        self._player = player - 1
        self._bv_rcc = bv_rcc
        self._prev_state = -1

    def _log(self, exp_sig):
        if self._bv_rcc is not None:
            if exp_sig != self._prev_state:
                self._bv_rcc.send_annotation(exp_sig)
                self._prev_state = exp_sig

    def run(self):
        import time
        while True:
            t = time.time()
            data = self._soc.recv(BUFFER_SIZE)
            game_time, p1_prog, p1_exp_sig, p2_prog, p2_exp_sig, p3_prog, p3_exp_sig, p4_prog, p4_exp_sig = unpack(
                'ffifififi', data)
            exp_sig = [p1_exp_sig, p2_exp_sig, p3_exp_sig, p4_exp_sig]
            exp_sig = exp_sig[self._player] % 10  # remove player number
            self._log(exp_sig)
            # print(time.time() - t)


def setup_logger(logger_name, log_file='', log_dir='log/', level=logging.INFO, log_to_stream=False):
    """Logger creation function.

    This function creates a logger, which has a separated log file, where it will append the logs.

    Parameters
    ----------
    logger_name : str
        Name of logger.
    log_file : str
        This string will be added to the filename. By default the filename contains
        the creation time and the .log extension
    log_dir : str
        The path where to save the .log files.
    level : logging const
        The level of log.
    log_to_stream : bool
        Log info to the stream also.

    """
    make_dir(log_dir)
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(name)s %(levelname)s: %(message)s')
    file_handler = logging.FileHandler(
        log_dir + '{}_{}.log'.format(log_file, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'), mode='a'))
    file_handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(file_handler)

    if log_to_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)


def log_info(logger_name, msg):
    """Logging info.

    If the logger was created before it will log the given message into it.

    Parameters
    ----------
    logger_name : str
        Logger name, which connects the info to the file.
    msg : str
        Log message

    """
    logger = logging.getLogger(logger_name)
    logger.info(msg)


if __name__ == '__main__':
    from brainvision import *
    rcc = RemoteControlClient()
    rcc.open_recorder()
    rcc.check_impedance()
    logger = GameLogger(rcc)
    logger.run()

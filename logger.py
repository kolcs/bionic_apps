import datetime
import logging
import socket
from enum import Enum
from struct import unpack
from threading import Thread

from preprocess import make_dir

UDP_IP = '127.0.0.1'
UDP_PORT = 8053
BUFFER_SIZE = 36

SESSION_START = 16
SESSION_END = 12

PLAYER = 'player'
PROGRESS = 'progress'
EXPECTED_SIG = 'exp_sig'

STATE = Enum('GameState', 'INIT RUN')


# trigger task converter is needed


class GameLogger(Thread):

    def __init__(self, bv_rcc=None, player=1, daemon=True):
        Thread.__init__(self, daemon=daemon)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((UDP_IP, UDP_PORT))
        self._player = player
        self._bv_rcc = bv_rcc
        self._prev_state = int()
        self._game_state = STATE.INIT
        self._players_state = tuple()
        self._init_game()

    def _init_game(self):
        # game_time, p1_prog, p1_exp_sig, p2_prog, p2_exp_sig, p3_prog, p3_exp_sig, p4_prog, p4_exp_sig
        self._prev_state = -1
        self._game_state = STATE.INIT
        self._players_state = (.0, .0, 0, .0, 0, .0, 0, .0, 0)

    def get_game_time(self):
        return self._players_state[0]

    def get_progress(self, player):
        assert player in range(1, 5), 'Player number should be between 1 and 4!'
        return self._players_state[player * 2 - 1]

    def get_expected_signal(self, player):
        assert player in range(1, 5), 'Player number should be between 1 and 4! {} were given'.format(player)
        return self._players_state[player * 2] % 10  # remove player number

    def _log_exp_sig(self, exp_sig):
        if self._game_state != STATE.INIT:
            make_log = False
            if exp_sig != self._prev_state:
                make_log = True
                self._prev_state = exp_sig
            if make_log:
                if self._bv_rcc is not None:
                    self._bv_rcc.send_annotation(exp_sig)
                else:
                    print(exp_sig)

    def log(self, msg):
        if self._game_state != STATE.INIT:
            if self._bv_rcc is not None:
                self._bv_rcc.send_annotation(msg)
            else:
                print(msg)

    def run(self):
        while True:
            try:
                data = self._sock.recv(BUFFER_SIZE)

                if self._game_state == STATE.INIT:
                    self._game_state = STATE.RUN
                    self._log_exp_sig(SESSION_START)
                    self._sock.settimeout(.5)

                # game_time, p1_prog, p1_exp_sig, p2_prog, p2_exp_sig, p3_prog, p3_exp_sig, p4_prog, p4_exp_sig
                self._players_state = unpack('ffifififi', data)
                exp_sig = self.get_expected_signal(self._player)
                self._log_exp_sig(exp_sig)

            except socket.timeout:
                self._log_exp_sig(SESSION_END)
                self._init_game()
                self._sock.settimeout(None)

    def __del__(self):
        del self._bv_rcc
        self._sock.close()


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
    # rcc = RemoteControlClient()
    # rcc.open_recorder()
    # rcc.check_impedance()
    logger = GameLogger(bv_rcc=None, daemon=False)
    logger.start()

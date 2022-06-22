import datetime
import logging
import socket
from enum import Enum
from pathlib import Path
from struct import unpack
from threading import Thread

from .commands import ControlCommand

UDP_IP = '127.0.0.1'
UDP_PORT = 8053
BUFFER_SIZE = 36

TRIGGER_FORMAT = 'S {:>2}'


class Trigger(Enum):
    SESSION_START = TRIGGER_FORMAT.format(16)
    SESSION_END = TRIGGER_FORMAT.format(12)
    ACTIVE = TRIGGER_FORMAT.format(5)
    CALM = TRIGGER_FORMAT.format(7)


PLAYER = 'player'
PROGRESS = 'progress'
EXPECTED_SIG = 'exp_sig'


class STATE(Enum):
    INIT = 1
    RUN = 2


def _build_cmd_trigger_conv(data_loader):
    if data_loader is None:
        return None
    cmd_trigger_conv = dict()
    trigger_task = data_loader.get_trigger_task_conv()
    for task, command in data_loader.get_command_converter().items():
        cmd_trigger_conv[command] = TRIGGER_FORMAT.format(trigger_task[task])
    return cmd_trigger_conv


class GameLogger(Thread):

    def __init__(self, bv_rcc=None, player=1, daemon=True, data_loader=None):
        Thread.__init__(self, daemon=daemon)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((UDP_IP, UDP_PORT))
        self._player = player
        self._bv_rcc = bv_rcc
        self._game_state = STATE.INIT
        self._prev_state = ''
        self._command_reached = bool()
        self._players_state = tuple()
        # self._players_prev_state = tuple()
        self._cmd_trigger_conv = _build_cmd_trigger_conv(data_loader)
        self._init_game()

    def _init_game(self):
        # game_time, p1_prog, p1_exp_sig, p2_prog, p2_exp_sig, p3_prog, p3_exp_sig, p4_prog, p4_exp_sig
        self._prev_state = ''
        self._game_state = STATE.INIT
        self._command_reached = False
        self._players_state = (.0, .0, 0, .0, 0, .0, 0, .0, 0)
        # self._players_prev_state = (.0, .0, 0, .0, 0, .0, 0, .0, 0)

    def get_game_time(self):
        return self._players_state[0]

    def get_progress(self, player):
        assert player in range(1, 5), 'Player number should be between 1 and 4!'
        return self._players_state[player * 2 - 1]

    def get_expected_signal(self, player):
        assert player in range(1, 5), 'Player number should be between 1 and 4! {} were given'.format(player)
        return self._players_state[player * 2] % 10  # remove player number

    # def get_speed(self, player):
    #     assert player in range(1, 5), 'Player number should be between 1 and 4! {} were given'.format(player)
    #     if self._game_state == STATE.INIT:
    #         return 0
    #     return (self._players_state[player * 2 - 1] - self._players_prev_state[player * 2 - 1]) / \
    #            (self._players_state[0] - self._players_prev_state[0])

    def _log_exp_sig(self, exp_sig):
        make_log = False
        if exp_sig != self._prev_state:
            make_log = True
            self._prev_state = exp_sig
        if make_log:
            self.log(exp_sig)

    def log(self, msg):
        if self._game_state != STATE.INIT:
            if self._bv_rcc is not None:
                self._bv_rcc.send_annotation(msg)
            else:
                print(msg)

    def log_toggle_switch(self, command):
        exp_sig = self.get_expected_signal(self._player)
        exp_cmd = ControlCommand(exp_sig)

        if exp_cmd == ControlCommand.STRAIGHT:
            self._command_reached = False
            if command == ControlCommand.STRAIGHT:
                self.log(Trigger.CALM.value)  # calm
            # else:
            #     print("Wrong command!")
        else:
            if exp_cmd == command:
                self._command_reached = True
                self.log(Trigger.ACTIVE.value)  # active
            elif self._command_reached:
                if command == ControlCommand.STRAIGHT:
                    self.log(Trigger.CALM.value)
                else:
                    self._command_reached = False
                    # print("Wrong command!")
            elif command != ControlCommand.STRAIGHT:
                self.log(Trigger.ACTIVE.value)
            # else:
            #     print("Wrong command!")

    def _log_track_changes(self):
        if self._cmd_trigger_conv is not None:
            exp_sig = self.get_expected_signal(self._player)
            exp_cmd = ControlCommand(exp_sig)
            self._log_exp_sig(self._cmd_trigger_conv[exp_cmd])

    def run(self):
        """ Thread function for self._player """
        while True:
            try:
                data = self._sock.recv(BUFFER_SIZE)

                if self._game_state == STATE.INIT:
                    self._game_state = STATE.RUN
                    self._log_exp_sig(Trigger.SESSION_START.value)
                    self._sock.settimeout(.5)

                # self._players_prev_state = self._players_state
                self._players_state = unpack('ffifififi', data)
                # exp_sig = self.get_expected_signal(self._player)
                # self._log_exp_sig(exp_sig)
                self._log_track_changes()

            except socket.timeout:
                self._log_exp_sig(Trigger.SESSION_END.value)
                self._init_game()
                self._sock.settimeout(None)

    def __del__(self):
        del self._bv_rcc
        self._sock.close()


def setup_logger(logger_name, log_to_stream=True, log_file=None, log_dir='log/',
                 verbose=True):
    """Logger creation function.

    This function creates a logger, which has a separated log file, where it will append the logs.

    Parameters
    ----------
    logger_name : str
        Name of logger.
    log_to_stream : bool
        Log info to the stream.
    log_file : str
        This string will be added to the filename. By default the filename contains
        the creation time and the .log extension
    log_dir : str
        The path where to save the .log files.
    verbose : bool
        The level of log. If true: info

    """
    level = logging.INFO if verbose else logging.WARNING
    Path(log_dir).mkdir(exist_ok=True)
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(name)s %(levelname)s: %(message)s')

    logger.setLevel(level)

    if log_file is not None:
        file_handler = logging.FileHandler(
            log_dir + '{}_{}.log'.format(log_file, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'), mode='a'))
        file_handler.setFormatter(formatter)
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

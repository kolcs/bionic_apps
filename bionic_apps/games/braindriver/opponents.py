from enum import Enum, auto
from multiprocessing import Pipe
from random import shuffle
from threading import Thread
from time import sleep

from numpy.random import randint

from .control import ControlCommand, GameControl
from .logger import GameLogger
from ...databases.eeg.defaults import ACTIVE


class PlayerType(Enum):
    MASTER = auto()
    RANDOM = auto()
    RANDOM_BINARY = auto()


class Player(GameControl, Thread):

    def __init__(self, player_num, game_log_conn, reaction_time=1.0, *, daemon=True):
        GameControl.__init__(self, player_num=player_num, game_log_conn=game_log_conn)
        Thread.__init__(self, daemon=daemon)
        self._reaction_time = reaction_time

    def _control_protocol(self):
        raise NotImplementedError('Control protocol not implemented')

    def run(self):
        while True:
            self._control_protocol()
            sleep(self._reaction_time)


class MasterPlayer(Player):

    def _control_protocol(self):
        assert self._game_log_conn is not None, 'GameLogger connection must be defined for MasterPlayer!'
        # exp_sig = self._game_logger.get_expected_signal(self.player_num)
        self._game_log_conn.send(['exp_sig', self.player_num])
        exp_sig = self._game_log_conn.recv()
        cmd = ControlCommand(exp_sig)
        self.control_game(cmd)


class RandomPlayer(Player):

    def __init__(self, player_num, reaction_time=1, *, daemon=True):
        super().__init__(player_num, None, reaction_time, daemon=daemon)

    def _control_protocol(self):
        cmd = ControlCommand(randint(4))
        self.control_game(cmd)


class RandomBinaryPlayer(RandomPlayer):

    def _control_protocol(self):
        cmd_list = [ACTIVE, 'none']
        cmd = cmd_list[randint(2)]
        self.control_game_with_2_opt(cmd)


def create_opponents(main_player=1, players=None, game_log_conn=None, reaction=1.0, *, daemon=True):
    """ Function for player creation.

    Creating required types of bot players.

    Parameters
    ----------
    main_player: int
        the number of main player
    players: list of string, list of PlayerType, None
        List of player types. Should contain 4 values.
    game_log_conn: multiprocessing.connection.Connection, None
        Optional predefined multiprocessing connection object
        to communicate between GameLogger and Players.
    reaction: float
        Player reaction time in seconds.
    """
    if players is None:
        players = [PlayerType.MASTER, PlayerType.RANDOM, PlayerType.RANDOM_BINARY]
        shuffle(players)
    assert len(players) == 3, '3 player type is required {} were given instead'.format(len(players))

    player_numbers = list(range(1, 5))
    player_numbers.remove(main_player)

    for num, pl_type in zip(player_numbers, players):
        if pl_type is PlayerType.MASTER:
            if game_log_conn is None:
                game_log_conn, child_conn = Pipe()
                GameLogger(daemon=daemon, connection=child_conn).start()
            bot = MasterPlayer(num, game_log_conn, daemon=daemon)
        elif pl_type is PlayerType.RANDOM:
            bot = RandomPlayer(num, reaction_time=reaction, daemon=daemon)
        elif pl_type is PlayerType.RANDOM_BINARY:
            bot = RandomBinaryPlayer(num, reaction_time=reaction, daemon=daemon)
        else:
            raise NotImplementedError('{} player is not implemented'.format(pl_type))
        bot.start()

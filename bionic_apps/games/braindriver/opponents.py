from enum import Enum, auto
from random import shuffle
from threading import Thread
from time import sleep

from config import ACTIVE, ControlCommand
from numpy.random import randint

from control import GameControl
from logger import GameLogger


class PlayerType(Enum):
    MASTER = auto()
    RANDOM = auto()
    RANDOM_BINARY = auto()


class Player(GameControl, Thread):

    def __init__(self, player_num, game_logger, reaction_time=1.0, daemon=True):
        GameControl.__init__(self, player_num=player_num, game_logger=game_logger)
        Thread.__init__(self, daemon=daemon)
        self._reaction_time = reaction_time

    def control_protocol(self):
        raise NotImplementedError('Control protocol not implemented')

    def run(self):
        while True:
            self.control_protocol()
            sleep(self._reaction_time)


class MasterPlayer(Player):

    def control_protocol(self):
        cmd = ControlCommand(self._game_logger.get_expected_signal(self.player_num))
        self.control_game(cmd)


class RandomPlayer(Player):

    def __init__(self, player_num, reaction_time=1, daemon=True):
        super().__init__(player_num, None, reaction_time, daemon)

    def control_protocol(self):
        cmd = ControlCommand(randint(4))
        self.control_game(cmd)


class RandomBinaryPlayer(RandomPlayer):

    def control_protocol(self):
        cmd_list = [ACTIVE, 'none']
        cmd = cmd_list[randint(2)]
        self.control_game_with_2_opt(cmd)


def create_opponents(main_player=1, players=None, game_logger=None, reaction=1.0):
    """ Function for player creation.

    Creating required type of bot players.

    Parameters
    ----------
    main_player: int
        the number of main player
    players: list of string, list of PlayerType, None
        List of player types. Should contain 4 values.
    game_logger: GameLogger, None
        Optional predefined GameLogger object
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
            if game_logger is None:
                game_logger = GameLogger()
                game_logger.start()
            bot = MasterPlayer(num, game_logger)
        elif pl_type is PlayerType.RANDOM:
            bot = RandomPlayer(num, reaction_time=reaction)
        elif pl_type is PlayerType.RANDOM_BINARY:
            bot = RandomBinaryPlayer(num, reaction_time=reaction)
        else:
            raise NotImplementedError('{} player is not implemented'.format(pl_type))
        bot.start()


if __name__ == '__main__':
    reaction = 2
    game_logger = GameLogger(daemon=False)
    game_logger.start()
    player2 = MasterPlayer(2, game_logger, daemon=False)
    player2.start()
    player3 = RandomPlayer(3, daemon=False, reaction_time=reaction)
    player3.start()
    player4 = RandomBinaryPlayer(4, daemon=False, reaction_time=reaction)
    player4.start()

from threading import Thread
from time import sleep

from numpy.random import randint

from config import ACTIVE, ControlCommand
from control import GameControl


class Player(GameControl, Thread):

    def __init__(self, player_num, game_logger, reaction_time=1, daemon=True):
        GameControl.__init__(self, player_num=player_num)
        Thread.__init__(self, daemon=daemon)
        self.game_logger = game_logger
        self._reaction_time = reaction_time

    def control_protocol(self):
        raise NotImplementedError('Control protocol not implemented')

    def run(self):
        while True:
            self.control_protocol()
            sleep(self._reaction_time)


class MasterPlayer(Player):

    def control_protocol(self):
        cmd = ControlCommand(self.game_logger.get_expected_signal(self.player_num))
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


if __name__ == '__main__':
    from logger import GameLogger

    reaction = 2
    game_logger = GameLogger(daemon=False)
    game_logger.start()
    player2 = MasterPlayer(2, game_logger, daemon=False)
    player2.start()
    player3 = RandomPlayer(3, daemon=False, reaction_time=reaction)
    player3.start()
    player4 = RandomBinaryPlayer(4, daemon=False, reaction_time=reaction)
    player4.start()

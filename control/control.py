import socket
from enum import Enum
from itertools import cycle
from threading import Thread
from time import sleep

from numpy import uint8
from numpy.random import randint

from config import TURN_LEFT, TURN_RIGHT, LIGHT_ON, GO_STRAIGHT, ACTIVE
from logger import setup_logger, log_info


class ControlCommand(Enum):
    LEFT = 1
    RIGHT = 3
    HEADLIGHT = 2
    STRAIGHT = 0


GAME_CONTROL_PORT = 5555
LOGGER_NAME = 'GameControl'


class GameControl(object):

    def __init__(self, player_num=1, udp_ip='localhost', udp_port=GAME_CONTROL_PORT,
                 make_log=False, log_to_stream=False):
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.player_num = player_num
        self._log = make_log
        self._command_menu = cycle([self.turn_left, self.turn_right, self.turn_light_on])
        if make_log:
            setup_logger(LOGGER_NAME, log_file='game', log_to_stream=log_to_stream)

    def _send_message(self, message):
        self.socket.sendto(message, (self.udp_ip, self.udp_port))

    def _log_message(self, message):
        if self._log:
            log_info(LOGGER_NAME, message)

    def turn_left(self):
        self._send_message(uint8(self.player_num * 10 + ControlCommand.LEFT.value))
        self._log_message('Command: Left turn')

    def turn_right(self):
        self._send_message(uint8(self.player_num * 10 + ControlCommand.RIGHT.value))
        self._log_message('Command: Right turn')

    def turn_light_on(self):
        self._send_message(uint8(self.player_num * 10 + ControlCommand.HEADLIGHT.value))
        self._log_message('Command: Light on')

    def go_straight(self):
        # do not sed anything because it will registered as wrong command...
        self._log_message('Command: Go straight')

    def game_started(self):
        self._log_message('Game started!')

    def control_game(self, command):
        if command == TURN_LEFT:
            self.turn_left()
        elif command == TURN_RIGHT:
            self.turn_right()
        elif command == LIGHT_ON:
            self.turn_light_on()
        elif command == GO_STRAIGHT:
            self.go_straight()
        else:
            raise NotImplementedError('Command {} is not implemented'.format(command))

    def control_game_with_2_opt(self, state):
        if state == ACTIVE:
            command = next(self._command_menu)
            command()
        else:
            self.go_straight()


class Player(GameControl, Thread):

    def __init__(self, player_num, game_logger, reaction_time=1, daemon=True):
        GameControl.__init__(self, player_num=player_num)
        Thread.__init__(self, daemon=daemon)
        # super(Process, self).__init__(daemon=daemon)
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
        if cmd == ControlCommand.HEADLIGHT:
            self.turn_light_on()
        elif cmd == ControlCommand.LEFT:
            self.turn_left()
        elif cmd == ControlCommand.RIGHT:
            self.turn_right()


class RandomPlayer(Player):

    def __init__(self, player_num, reaction_time=1, daemon=True):
        super().__init__(player_num, None, reaction_time, daemon)

    def control_protocol(self):
        cmd = ControlCommand(randint(4))
        if cmd == ControlCommand.HEADLIGHT:
            self.turn_light_on()
        elif cmd == ControlCommand.LEFT:
            self.turn_left()
        elif cmd == ControlCommand.RIGHT:
            self.turn_right()


class RandomBinaryPlayer(Player):

    def __init__(self, player_num, reaction_time=1, daemon=True):
        super().__init__(player_num, None, reaction_time, daemon)

    def control_protocol(self):
        cmd_list = [ACTIVE, 'none']
        cmd = cmd_list[randint(2)]
        self.control_game_with_2_opt(cmd)


def run_demo(make_log=False):
    from pynput.keyboard import Key, Listener

    controller = GameControl(make_log=make_log)

    def control(key):
        if key == Key.up:
            controller.turn_light_on()
        elif key == Key.left:
            controller.turn_left()
        elif key == Key.right:
            controller.turn_right()

    def on_press(key):
        control(key)
        print('{0} pressed'.format(
            key))

    def on_release(key):
        # print('{0} release'.format(
        #     key))
        if key == Key.esc:
            # Stop listener
            return False

    # Collect events until released
    with Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()


if __name__ == '__main__':
    run_demo()

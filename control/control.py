import socket
import json
from numpy import uint8

from config import TURN_LEFT, TURN_RIGHT, LIGHT_ON, GO_STRAIGHT
from logger import *

LEFT = 1
RIGHT = 3
HEADLIGHT = 2
STRAIGHT = 0

GAME_CONTROL_PORT = 5555

LOGGER_NAME = 'GameControl'


# todo: check from where to load data... networkConfig.json or CONST PARAMS?


class GameControl(object):

    def __init__(self, path='control/', make_log=False):
        with open(path + 'networkConfig.json') as f:
            config = json.load(f)
        self.udp_ip = config['networkAddress']
        self.udp_port = GAME_CONTROL_PORT
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.player_index = config['activPlayerIndex'] * 10
        self._log = make_log
        if make_log:
            setup_logger(LOGGER_NAME, log_file='game')

    def _send_message(self, message):
        self.socket.sendto(message, (self.udp_ip, self.udp_port))

    def _log_message(self, message):
        if self._log:
            log_info(LOGGER_NAME, message)

    def turn_left(self):
        self._send_message(uint8(self.player_index + LEFT))
        self._log_message('Command: Left turn')

    def turn_right(self):
        self._send_message(uint8(self.player_index + RIGHT))
        self._log_message('Command: Right turn')

    def turn_light_on(self):
        self._send_message(uint8(self.player_index + HEADLIGHT))
        self._log_message('Command: Light on')

    def go_straight(self):
        # do not sed anything because it will registered as wrong command...
        # self._send_message(uint8(self.player_index + STRAIGHT))
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


def run_demo(make_log=False):
    from pynput.keyboard import Key, Listener

    controller = GameControl(path='', make_log=make_log)

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

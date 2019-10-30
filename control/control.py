import socket
import json
from numpy import uint8
import logging
import datetime
from time import time

from config import TURN_LEFT, TURN_RIGHT, LIGHT_ON, GO_STRAIGHT
from preprocess import make_dir

LEFT = 1
RIGHT = 3
HEADLIGHT = 2
STRAIGHT = 0

GAME_CONTROL_PORT = 5555


def _init_log(log):
    if log:
        make_dir('log')
        return logging.basicConfig(filename='log/game_{}.log'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')),
                                   filemode='a',
                                   format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                   datefmt='%H:%M:%S',
                                   level=logging.DEBUG)
    else:
        return None


# todo: check from where to load data... networkConfig.json or CONST PARAMS?


class GameControl(object):

    def __init__(self, path='', log=False):
        with open(path + 'networkConfig.json') as f:
            config = json.load(f)
        self.udp_ip = config['networkAddress']
        self.udp_port = GAME_CONTROL_PORT
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.player_index = config['activPlayerIndex'] * 10
        self._logger = _init_log(log)

    def _send_message(self, message):
        self.socket.sendto(message, (self.udp_ip, self.udp_port))

    def turn_left(self):
        self._send_message(uint8(self.player_index + LEFT))

    def turn_right(self):
        self._send_message(uint8(self.player_index + RIGHT))

    def turn_light_on(self):
        self._send_message(uint8(self.player_index + HEADLIGHT))

    # def go_straight(self):
    #     self._send_message(uint8(self.player_index + STRAIGHT))

    def game_started(self):
        if self._logger is not None:
            self._logger.info('brainDriver started at {}'.format(time()))

    def control_game(self, command):
        if command == TURN_LEFT:
            self.turn_left()
        elif command == TURN_RIGHT:
            self.turn_right()
        elif command == LIGHT_ON:
            self.turn_light_on()
        elif command == GO_STRAIGHT:
            pass
        else:
            raise NotImplementedError('Command {} is not implemented'.format(command))


def run_demo():
    from pynput.keyboard import Key, Listener

    controller = GameControl()

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

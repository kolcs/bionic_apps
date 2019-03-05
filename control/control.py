import socket
import json
from numpy import uint8

LEFT = 1
RIGHT = 3
HEADLIGHT = 2

GAME_CONTROL_PORT = 5555


class GameControl(object):

    def __init__(self, path=''):
        with open(path + 'networkConfig.json') as f:
            config = json.load(f)
        self.udp_ip = config['networkAddress']
        self.udp_port = GAME_CONTROL_PORT
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.player_index = config['activPlayerIndex'] * 10

    def _send_message(self, message):
        self.socket.sendto(message, (self.udp_ip, self.udp_port))

    def turn_left(self):
        self._send_message(uint8(self.player_index + LEFT))

    def turn_right(self):
        self._send_message(uint8(self.player_index + RIGHT))

    def turn_light_on(self):
        self._send_message(uint8(self.player_index + HEADLIGHT))

    def run_demo(self):
        from pynput.keyboard import Key, Listener

        def control(key):
            if key == Key.up:
                self.turn_light_on()
            elif key == Key.left:
                self.turn_left()
            elif key == Key.right:
                self.turn_right()

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
    GameControl().run_demo()

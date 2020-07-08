import socket
import threading
from os import startfile
from time import sleep

TCP_IP = '127.0.0.1'
TCP_PORT = 6700
BUFFER_SIZE = 1024

REMOTE_CONTROL_SERVER_PATH = r'C:\Vision\RemoteControlServer\RemoteControlServer.exe'

APPLICATION_SATE = 'AP'
RECORDER_STATE = 'RS'
ACQUISITION_SATE = 'AQ'

ANS_WAITING_TIME = 0.1  # sec


class RemoteControlClient(object):
    def __init__(self, start_remote_control_server=True, print_received_messages=True):
        if start_remote_control_server:
            startfile(REMOTE_CONTROL_SERVER_PATH)

        self._print_answers = print_received_messages

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((TCP_IP, TCP_PORT))
        self._sent_msg = str()
        self._state = {APPLICATION_SATE: str(), RECORDER_STATE: str(), ACQUISITION_SATE: str()}

        self._listen_thread = threading.Thread(target=self._listening_message_in_thread, daemon=True)
        self._listen_thread.start()
        self.ask_msg_protocol()

    def _send_message(self, msg, required_state=None, code=None):
        self._waiting_for_required_state(required_state, code)
        self._sent_msg = msg
        msg += '\r'
        self._sock.send(msg.encode())

    def _waiting_for_required_state(self, state=None, code=None):
        if state is not None:
            code = str(code)
            while self._state[state] != code:
                sleep(ANS_WAITING_TIME)

    def _get_message(self):
        return self._sock.recv(BUFFER_SIZE).decode().strip('\r')

    def _listening_message_in_thread(self):
        while True:
            msg = self._get_message()
            ans = msg.split(':')

            if ans[0] == self._sent_msg:
                self._sent_msg = str()
                if ans[1] == 'Error':
                    print(ans[2])
                if ans[0] == 'VM':
                    assert ans[1] == '2', 'Only Messaging protocol 2 is supported!'

            if ans[0] in self._state.keys():
                self._state[ans[0]] = ans[1]
                # print(ans[0], self._state[ans[0]], ans[1])

            if self._print_answers:
                print(msg)

    def __del__(self):
        self.stop_rec()
        self.close_recorder()
        self._sock.close()

    def ask_msg_protocol(self):
        self._send_message('VM')

    def open_recorder(self):
        self._send_message('O')

    def view_data(self):
        self._send_message('M')

    def check_impedance(self):
        self._send_message('I', APPLICATION_SATE, 1)

    def test_signal(self):
        self._send_message('T', APPLICATION_SATE, 1)

    def stop_view(self):
        self._send_message('SV')

    def start_rec(self):
        self._send_message('S')

    def pause_rec(self):
        self._send_message('P')

    def resume_rec(self):
        self._send_message('C')

    def stop_rec(self):
        self._send_message('Q')

    def close_recorder(self):
        self._send_message('X')

    def reset_DC(self):
        self._send_message('D')

    def request_recorder_state(self):
        self._send_message(RECORDER_STATE)

    def request_application_state(self):
        self._send_message(APPLICATION_SATE)

    def request_acquisition_state(self):
        self._send_message(ACQUISITION_SATE)

    def send_annotation(self, annotation, ann_type='Stimulus'):
        self._send_message('AN:{};{}'.format(annotation, ann_type))


if __name__ == '__main__':
    import time

    rcc = RemoteControlClient()
    rcc.open_recorder()
    rcc.check_impedance()
    time.sleep(10)

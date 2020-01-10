import socket
import threading

TCP_IP = '127.0.0.1'
TCP_PORT = 6700
BUFFER_SIZE = 1024


class RemoteControlClient(object):
    def __init__(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((TCP_IP, TCP_PORT))
        self._sent_msg = str()
        self._state = {'AP': str(), 'RS': str(), 'AQ': str()}
        self._listen_thread = threading.Thread(target=self._listening_message_in_thread, daemon=True)
        self._listen_thread.start()

    def _send_message(self, msg):
        self._sent_msg = msg
        msg += '\r'
        self._sock.send(msg.encode())

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

            print(msg)

    def __del__(self):
        self._sock.close()

    def ask_msg_protocol(self):
        self._send_message('VM')

    def open_recorder(self):
        self._send_message('O')

    def view_data(self):
        self._send_message('M')

    def check_impedance(self):
        self._send_message('I')

    def test_signal(self):
        self._send_message('T')

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
        self._send_message('RS')

    def request_application_state(self):
        self._send_message('AP')

    def request_acquisition_state(self):
        self._send_message('AQ')


if __name__ == '__main__':
    import time
    rcc = RemoteControlClient()
    rcc.ask_msg_protocol()
    rcc.open_recorder()
    time.sleep(10)

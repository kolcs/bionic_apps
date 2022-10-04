from multiprocessing import Pipe

from bionic_apps.databases import Databases
from bionic_apps.games.braindriver.logger import GameLogger
from bionic_apps.games.braindriver.opponents import MasterPlayer
from bionic_apps.preprocess.io import DataLoader


def run_braindriver_paradigm(db_name=Databases.PILOT_PAR_B):
    loader = DataLoader()
    loader.use_db(db_name)

    # from bionic_apps.external_connections.brainvision import RemoteControlClient
    # rcc = RemoteControlClient(print_received_messages=False)
    # rcc.open_recorder()
    # rcc.view_data()
    print('Connecting to game...')
    parent_conn, child_conn = Pipe()

    game_log = GameLogger(data_loader=loader, connection=child_conn, annotator='bv_rcc')
    # game_log = GameLogger(data_loader=loader, connection=child_conn, annotator=None)
    game_log.start()
    MasterPlayer(player_num=1, game_log_conn=parent_conn, reaction_time=.5).start()

    while True:
        pass


if __name__ == '__main__':
    run_braindriver_paradigm()

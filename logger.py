import logging
import datetime

from preprocess import make_dir


def setup_logger(logger_name, log_file='', log_dir='log/', level=logging.INFO, log_to_stream=False):
    make_dir(log_dir)
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(name)s %(levelname)s: %(message)s')
    file_handler = logging.FileHandler(
        log_dir + '{}_{}.log'.format(log_file, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'), mode='a'))
    file_handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(file_handler)

    if log_to_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)


def log_info(logger_name, msg):
    logger = logging.getLogger(logger_name)
    logger.info(msg)

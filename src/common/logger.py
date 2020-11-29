import os
import logging
from logging.handlers import RotatingFileHandler

logging.getLogger().setLevel(logging.INFO)


def get_formatter():
    formatter = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
    return formatter


def get_local_file_handler(formatter):
    os.makedirs('tmp/', exist_ok = True)
    file_handler = RotatingFileHandler(
        'tmp/log.txt', maxBytes=(20 * 1000 * 1000), backupCount=20
    )
    file_handler.setFormatter(logging.Formatter(formatter))
    return file_handler


def get_console_handler(formatter):
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(formatter))
    return handler


def get_logger(logger_name = 'log'):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = get_formatter()
    logger.addHandler(get_local_file_handler(formatter))
    logger.addHandler(get_console_handler(formatter))
    return logger
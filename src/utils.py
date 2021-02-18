from time import gmtime, strftime
import os
import json
import logging


def create_training_outputs_folder(path2store):
    folder_name = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    folder_path = os.path.join(path2store, folder_name)

    if os.path.isdir(folder_path) is False:
        os.mkdir(folder_path)

    return folder_path


def save_to_json(json_name, data):
    with open(json_name, 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4)


def convert_to_preferred_format(sec):
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60
    return "%02d:%02d:%02d" % (hour, min, sec)


def create_logger(logfile, level='NOTSET'):

    level_converter = {
        'NOTSET': logging.NOTSET,
        'DEBUG': logging.DEBUG,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }

    logger = logging.getLogger('my_logger')
    logger.setLevel(level_converter[level])

    # our first handler is a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.NOTSET)
    console_handler_format = '%(asctime)s | %(levelname)s: %(message)s'
    console_handler.setFormatter(logging.Formatter(console_handler_format))
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(level_converter[level])
    file_handler_format = '%(asctime)s | %(levelname)s: %(message)s'
    file_handler.setFormatter(logging.Formatter(file_handler_format))
    logger.addHandler(file_handler)

    return logger

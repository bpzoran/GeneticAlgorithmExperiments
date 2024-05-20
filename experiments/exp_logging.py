import datetime
import logging
import os
from os.path import isfile, join

from gadapt.utils.TimeStampFormatter import TimestampFormatter


def init_logging(log_to_file: bool):
    """
    Initializes logging for genetic algorithm
    """

    # Get platform-specific path separator
    path_separator = os.path.sep

    path = os.path.join(os.getcwd(), "log")
    if not os.path.exists(path):
        os.mkdir(path)
    now = datetime.datetime.now()
    formatted_date_time = now.strftime('%Y_%m_%d_%H_%M_%S_') + f'{now.microsecond // 1000:03d}'
    log_path = os.path.join(path, f"ga_exp_log_{formatted_date_time}.log")
    logger = logging.getLogger("ga_exp_logger")
    for h in logger.handlers:
        logger.removeHandler(h)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        TimestampFormatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(console_handler)
    if log_to_file:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            TimestampFormatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


def log_message_info(message: str):
    ga_exp_logger = logging.getLogger("ga_exp_logger")
    ga_exp_logger.log(level=logging.INFO, msg=message)

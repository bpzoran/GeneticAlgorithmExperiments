import logging
import os
from os.path import isfile, join

from gadapt.utils.TimeStampFormatter import TimestampFormatter


def init_logging(log_to_file: bool):
    """
    Initializes logging for genetic algorithm
    """

    def get_last_num(s: str) -> int:
        try:
            if not s.startswith("ga_exp_log.log"):
                return -1
            if s == "ga_exp_log.log":
                return 0
            s_last_number = s.rsplit(".", 1)[-1]
            n_last_number = int(s_last_number)
            s.rsplit(".", 1)[-1]
            return n_last_number
        except Exception:
            return -1

    # Get platform-specific path separator
    path_separator = os.path.sep

    path = os.path.join(os.getcwd(), "log")
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        onlyfiles = [f for f in os.listdir(path) if isfile(join(path, f))]
        onlyfiles.sort(reverse=True, key=get_last_num)
        for f in onlyfiles:
            if f == "ga_exp_log.log":
                try:
                    os.rename(os.path.join(path, f), os.path.join(path, "ga_exp_log.log.1"))
                except Exception:
                    print("Unable to rename log file: " + os.path.join(path, f))
                    break
            elif f.startswith("ga_exp_log.log."):
                n_last_number = get_last_num(f)
                if n_last_number == -1:
                    continue
                n_last_number += 1
                try:
                    os.rename(os.path.join(path, f), os.path.join(path, "ga_exp_log.log." + str(n_last_number)))
                except Exception:
                    break
    logpath = os.path.join(path, "ga_exp_log.log")
    logger = logging.getLogger("ga_exp_logger")
    # Remove all handlers associated with the logger object
    for h in logger.handlers:
        logger.removeHandler(h)

    # File handler

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        TimestampFormatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(console_handler)
    if log_to_file:
        file_handler = logging.FileHandler(logpath)
        file_handler.setFormatter(
            TimestampFormatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


def log_message_info(message: str):
    ga_exp_logger = logging.getLogger("ga_exp_logger")
    ga_exp_logger.log(level=logging.INFO, msg=message)

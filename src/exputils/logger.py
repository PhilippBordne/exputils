import sys
import logging


def setup_logger(logger: logging.Logger) -> logging.Logger:
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

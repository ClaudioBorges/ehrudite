import logging


def config_logging(verbose_level=0):
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(len(levels) - 1, verbose_level)]  # capped to number of levels
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")

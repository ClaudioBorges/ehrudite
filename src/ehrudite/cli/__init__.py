from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
import logging


def make_parser():
    parser = ArgumentParser(
        description="A deep learning (DL) framework for electronic health records (EHR)",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help=r"increse output verbosity",
    )
    return parser


def set_logging_level(verbose):
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(len(levels) - 1, verbose)]  # capped to number of levels
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def cli():
    parser = make_parser()
    args = parser.parse_args()
    set_logging_level(args.verbose)

    # Nothing to add
    logging.info("Finished")

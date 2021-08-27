from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
import ehrudite.cli.base as cli_base
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


def cli():
    parser = make_parser()
    args = parser.parse_args()
    cli_base.config_logging(args.verbose)

    logging.info("Started")
    logging.info("Finished")

from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
import ehrudite.cli.base as cli_base
import logging


def make_parser():
    parser = ArgumentParser(
        description="A deep learning (DL) framework for electronic health records (EHR)"
        + "\nUse: "
        + "\n  ehrudite-pip   - to run a DL pipeline"
        + "\n  ehrudite-stat  - to create statistic"
        + "\n  ehrudite-tok   - to tokenize an input",
        formatter_class=RawTextHelpFormatter,
    )
    return parser


def cli():
    parser = make_parser()
    args = parser.parse_args()

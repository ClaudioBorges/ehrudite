from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
import ehrudite.cli.base as cli_base
import ehrudite.core.statistic as stat
import logging


def make_parser():
    parser = ArgumentParser(
        description="Statistic for Ehrudite",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="../",
        help=r"default output artifacts location",
    )
    parser.add_argument(
        "-w",
        "--wordpiece_file",
        action="store",
        help=r"wordpiece vocab file",
    )
    parser.add_argument(
        "-s",
        "--sentencepiece_file",
        action="store",
        help=r"sentence model file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help=r"increse output verbosity",
    )
    parser.add_argument("ehrpreper_file", help="the ehrpreper file")
    return parser


def cli():
    parser = make_parser()
    args = parser.parse_args()
    cli_base.config_logging(args.verbose)

    logging.info(
        f"Started (ehrpreper_file={args.ehrpreper_file}, output_path={args.output_path})"
    )
    stat.from_ehrpreper(args.ehrpreper_file, args.output_path)
    if args.wordpiece_file is not None:
        stat.from_wordpiece_ehrpreper(
            args.ehrpreper_file, args.wordpiece_file, args.output_path
        )
    if args.sentencepiece_file is not None:
        stat.from_sentencepiece_ehrpreper(
            args.ehrpreper_file, args.sentencepiece_file, args.output_path
        )

    logging.info("Finished")

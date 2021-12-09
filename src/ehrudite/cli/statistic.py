from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
import ehrudite.cli.base as cli_base
import ehrudite.core.pipeline as pip
import ehrudite.core.pipeline.tokenizer as pip_tok
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
        "--wordpiece",
        action="store_true",
        help=r"statistic for wordpiece",
    )
    parser.add_argument(
        "-s",
        "--sentencepiece",
        action="store_true",
        help=r"statistics for sentencepiece",
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

    if args.wordpiece:
        logging.info(f"Wordpiece statistic...")
        stat.from_tokenizer_ehrpreper(
            args.ehrpreper_file, pip_tok.TokenizerType.WORDPIECE, args.output_path
        )
    if args.sentencepiece:
        logging.info(f"Sentencepiece statistic...")
        stat.from_tokenizer_ehrpreper(
            args.ehrpreper_file, pip_tok.TokenizerType.SENTENCEPIECE, args.output_path
        )

    logging.info("Finished")

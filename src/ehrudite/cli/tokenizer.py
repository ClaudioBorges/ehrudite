from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from ehrudite.core.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer
import ehrudite.cli.base as cli_base
import ehrudite.core.statistic as stat
import logging

SENTENCE_PIECE = "sentencepiece"
METHOD_CHOICES = [
    SENTENCE_PIECE,
]


def make_parser():
    parser = ArgumentParser(
        description="Statistic for Ehrudite",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-g",
        "--generate_vocab",
        action="store_true",
        help=r"generate vocabulary from a ehrpreper",
    )
    parser.add_argument(
        "-e",
        "--ehrpreper_file",
        action="store",
        help=r"the ehrpreper file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help=r"increse output verbosity",
    )
    parser.add_argument("vocab_path", help="vocabulary artefact path")
    parser.add_argument("method", choices=METHOD_CHOICES, help="tokenization method")
    return parser


def cli():
    parser = make_parser()
    args = parser.parse_args()
    cli_base.config_logging(args.verbose)

    logging.info(f"Started (method={args.method})")
    if args.method == SENTENCE_PIECE:
        if args.generate_vocab:
            logging.info(
                f"Generating vocabulary (ehrpreper_file={args.ehrpreper_file}, vocab_path={args.vocab_path})"
            )
            SentencePieceTokenizer.generate_vocab(
                [args.ehrpreper_file], args.vocab_path
            )

    logging.info("Finished")

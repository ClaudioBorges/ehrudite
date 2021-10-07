from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from ehrudite.core.pipeline import ehrpreper_k_fold_gen
import ehrudite.cli.base as cli_base
import ehrudite.core.pipeline.dnn as pip_dnn
import ehrudite.core.pipeline.tokenizer as pip_tok
import logging
import os


EHRPREPER_FILENAME = "ehrpreper.xml"
EHR_DATA = "../ehr-data"
N_SPLITS = 4


def make_parser():
    parser = ArgumentParser(
        description="Ehrudite pipelines",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-pt",
        "--pipeline_tokenizer",
        action="store_true",
        help=r"train tokenizer from ehrpreper",
    )
    parser.add_argument(
        "-pxx",
        "--pipeline_xfmr_xfmr",
        action="store_true",
        help=r"train transformer-transformer",
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help=r"run tests",
    )
    parser.add_argument(
        "-n",
        "--n_splits",
        action="store",
        default=N_SPLITS,
        help=f"number k fold splits (default={N_SPLITS})",
    )
    parser.add_argument(
        "-d",
        "--ehr_data_path",
        action="store",
        default=EHR_DATA,
        help=f'ehr-data base path (default="{EHR_DATA}")',
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

    logging.info(
        f"Started (n_splits={args.n_splits}, ehr_data_path={args.ehr_data_path})"
    )

    ehrpreper_file = os.path.join(args.ehr_data_path, EHRPREPER_FILENAME)
    k_folded = ehrpreper_k_fold_gen(ehrpreper_file, n_splits=args.n_splits)

    for run_id, (
        train_xy,
        test_xy,
    ) in enumerate(k_folded):
        logging.info(f"Running k_fold (run_id={run_id})")
        # Pipeline Tokenizer
        for tokenizer_type in pip_tok.TokenizerType:
            if args.pipeline_tokenizer:
                pip_tok.train(
                    run_id,
                    tokenizer_type,
                    train_xy,
                )
            if args.pipeline_xfmr_xfmr:
                if args.test:
                    run_tests(run_id, tokenizer_type, train_xy, test_xy)
                    return
                else:
                    pip_dnn.train_xfmr_xfmr(
                        run_id,
                        tokenizer_type,
                        train_xy,
                        test_xy,
                    )

    logging.info("Finished")


def run_tests(run_id, tokenizer_type, train_xy, test_xy):
    translator = pip_dnn.Translator(run_id, tokenizer_type)
    for (i, (x, y)) in enumerate(train_xy):
        print("-" * 80)
        print("START")
        print(f"Iteration (i={i})")
        # print(f"Input X={x}")
        print(f"Input Y={y}")
        translated_text, translated_tokens, attention_weights = translator(x, y)
        print(f"text={translated_text}, tokens={translated_tokens}")
        print("END")
    return

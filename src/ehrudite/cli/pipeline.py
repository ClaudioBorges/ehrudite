from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
import ehrudite.cli.base as cli_base
import ehrudite.core.pipeline as pip
import ehrudite.core.pipeline.dnn as pip_dnn
import ehrudite.core.pipeline.tokenizer as pip_tok
import logging
import os


EHRPREPER_FILENAME = "ehrpreper.xml"
EHR_DATA = "../ehr-data"
N_SPLITS = 4


TOKENIZER_ALLOW_LIST = [
    pip_tok.TokenizerType.SENTENCEPIECE,
    # pip_tok.TokenizerType.WORDPIECE,
]


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
        "-pll",
        "--pipeline_lstm_lstm",
        action="store_true",
        help=r"train lstm-lstm",
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help=r"run tests and validation",
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
    k_folded = pip.ehrpreper_k_fold_gen(ehrpreper_file, n_splits=args.n_splits)

    for run_id, (
        train_xy,
        test_xy,
    ) in enumerate(k_folded):
        logging.info(f"Running k_fold (run_id={run_id})")
        # Run for each tokenizer
        for tokenizer_type in TOKENIZER_ALLOW_LIST:

            def run_dnn_pipeline(dnn_type, n_epochs=[0, 0, 0, 265]):
                if args.test:
                    pip_dnn.validate(
                        run_id,
                        dnn_type,
                        tokenizer_type,
                        train_xy,
                        test_xy,
                    )
                else:
                    pip_dnn.train(
                        run_id,
                        dnn_type,
                        tokenizer_type,
                        train_xy,
                        test_xy,
                        n_epochs=n_epochs[run_id],
                    )

            # Tokenizer
            if args.pipeline_tokenizer:
                pip_tok.train(
                    run_id,
                    tokenizer_type,
                    train_xy,
                )
            # DNN Training and validate - XFMR-XFMR
            elif args.pipeline_xfmr_xfmr:
                run_dnn_pipeline(pip_dnn.DnnType.XFMR_XFMR)
            # DNN Training and validate - LSTM-LSTM
            elif args.pipeline_lstm_lstm:
                run_dnn_pipeline(pip_dnn.DnnType.LSTM_LSTM)

    logging.info("Finished")

"""ehrudite tokenizer pipeline"""


from ehrudite.core.pipeline import BASE_PATH
from ehrudite.core.pipeline import make_progressable
from ehrudite.core.pipeline import unpack_2d
from enum import Enum
import ehrudite.core.text as er_text
import ehrudite.core.tokenizer.sentencepiece_tokenizer as sentencepiece
import ehrudite.core.tokenizer.wordpiece_tokenizer as wordpiece
import logging
import os


TOKENIZER_BASE_PATH = os.path.join(BASE_PATH, "tokenizer/")

VOCAB_SIZE_X = 2 ** 14
VOCAB_SIZE_Y = 2 ** 9

PAD_TOK = 0
BOS_TOK = 1
EOS_TOK = 2
UNK_TOK = 3


class TokenizerType(Enum):
    SENTENCEPIECE = 1
    WORDPIECE = 2


SPECIFICATIONS = {
    TokenizerType.SENTENCEPIECE: {
        "base_ckpt": os.path.join(TOKENIZER_BASE_PATH, "sentencepiece/"),
        "class": sentencepiece.SentencepieceTokenizer,
        "gen_vocab": sentencepiece.generate_vocab_from_texts,
    },
    TokenizerType.WORDPIECE: {
        "base_ckpt": os.path.join(TOKENIZER_BASE_PATH, "wordpiece/"),
        "class": wordpiece.WordpieceTokenizer,
        "gen_vocab": wordpiece.generate_vocab_from_texts,
    },
}


def _get_specification(tokenizer_type):
    return SPECIFICATIONS[tokenizer_type]


def _get_train_checkpoints(run_id, specification, base_ckpt=None):
    base = base_ckpt or specification["base_ckpt"]
    train_x_id = f"train_x_{run_id}"
    train_y_id = f"train_y_{run_id}"
    return (
        os.path.join(base, train_x_id),
        os.path.join(base, train_y_id),
    )


def restore(run_id, tokenizer_type, **kwargs):
    logging.info(
        f"Restoring tokenizers (run_id={run_id}, tokenizer_type={tokenizer_type})"
    )
    spec = _get_specification(tokenizer_type)
    ckpt_x, ckpt_y = _get_train_checkpoints(run_id, spec, **kwargs)
    return (
        spec["class"](ckpt_x),
        spec["class"](ckpt_y),
    )


def train(
    run_id,
    tokenizer_type,
    train_xy,
    vocab_size_x=VOCAB_SIZE_X,
    vocab_size_y=VOCAB_SIZE_Y,
    **kwargs,
):
    logging.info(
        f"Training tokenizers (run_id={run_id}, tokenizer_type={tokenizer_type})"
    )
    train_x, train_y = unpack_2d(train_xy)
    train_x = make_progressable(train_x)
    train_y = make_progressable(train_y)

    spec = _get_specification(tokenizer_type)
    ckpt_x, ckpt_y = _get_train_checkpoints(run_id, spec, **kwargs)

    gen_vocab = spec["gen_vocab"]
    logging.debug(
        f"Training tokenizer for X (run_id={run_id}, "
        + f"tokenizer_type={tokenizer_type}, size={VOCAB_SIZE_X})"
    )
    gen_vocab(train_x, ckpt_x, vocab_size=vocab_size_x)
    logging.debug(
        f"Training tokenizer for Y (run_id={run_id}, "
        + f"tokenizer_type={tokenizer_type}, size={VOCAB_SIZE_Y})"
    )
    gen_vocab(train_y, ckpt_y, vocab_size=vocab_size_y)

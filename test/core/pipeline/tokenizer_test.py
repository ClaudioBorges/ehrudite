"""The test file for sentencepiece tokenizer"""

import ehrudite.core.pipeline.tokenizer as pip_tok
import ehrudite.core.text as er_text
import random
import tempfile


def test_tokenizer_pipeline():
    for tokenizer_type in pip_tok.TokenizerType:
        _test_tokenizer_pipeline(tokenizer_type)


def _test_tokenizer_pipeline(tokenizer_type):
    with tempfile.TemporaryDirectory() as dir_name:
        run_id = random.randint(1, 2 ** 32)
        train_xy = (
            (
                "To be, or not to be, that is the question.",
                "Sentence 1",
            ),
            (
                "Wisely and slow; they stumble that run fast.",
                "Sentence 2",
            ),
            (
                "Some Cupid kills with arrows, some with traps.",
                "Sentence 2",
            ),
        )

        pip_tok.train(
            run_id,
            tokenizer_type,
            train_xy,
            vocab_size_x=32,
            vocab_size_y=12,
            base_ckpt=dir_name,
        )

        tokenizers = pip_tok.restore(run_id, tokenizer_type, base_ckpt=dir_name)
        assert len(tokenizers) == 2
        for tokenizer in tokenizers:
            tokens = tokenizer.tokenize("random")
            assert tokens is not None
            assert len(tokens) >= 0

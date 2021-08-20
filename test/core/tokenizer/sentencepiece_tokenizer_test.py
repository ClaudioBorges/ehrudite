"""The test file for sentencepiece tokenizer"""

from ehrudite.core.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer
import tempfile


MOCKED_EHRPREPER = "test-data/ehrprepered.xml"
VOCABULARY_SIZE = 128


def test_tokenizer_detokenizer():
    with tempfile.TemporaryDirectory() as dir_name:
        output_prefix = f"{dir_name}/m"
        SentencePieceTokenizer.generate_vocab(
            [MOCKED_EHRPREPER], output_prefix, VOCABULARY_SIZE
        )
        model_file = f"{output_prefix}.model"
        tokenizer = SentencePieceTokenizer(model_file)

        sentences = [
            "To be, or not to be, that is the question.",
            "Wisely and slow; they stumble that run fast.",
            "Some Cupid kills with arrows, some with traps.",
        ]
        for sentence in sentences:
            tokens = tokenizer.tokenize(sentence)
            detokens = tokenizer.detokenize(tokens)
            assert sentence == detokens

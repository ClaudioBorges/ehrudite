"""The test file for sentencepiece tokenizer"""

import ehrudite.core.text as er_text
import ehrudite.core.tokenizer.sentencepiece_tokenizer as sentencepiece
import tempfile


MOCKED_EHRPREPER = "test-data/ehrprepered.xml"
VOCABULARY_SIZE = 128


def test_tokenizer_detokenizer():
    with tempfile.TemporaryDirectory() as dir_name:
        vocab_file = f"{dir_name}/vocab.txt"
        sentencepiece.generate_vocab([MOCKED_EHRPREPER], vocab_file, VOCABULARY_SIZE)
        tokenizer = sentencepiece.SentencepieceTokenizer(vocab_file)

        assert (
            VOCABULARY_SIZE * 0.7 <= tokenizer.vocab_size().numpy() <= VOCABULARY_SIZE
        )

        sentences = [
            "To be, or not to be, that is the question.",
            "Wisely and slow; they stumble that run fast.",
            "Some Cupid kills with arrows, some with traps.",
        ]
        preprocessed = er_text.preprocess(sentences)
        for sentence in preprocessed:
            tokens = tokenizer.tokenize(sentence)
            detokens = tokenizer.detokenize(tokens)
            assert sentence.encode() == detokens.numpy()

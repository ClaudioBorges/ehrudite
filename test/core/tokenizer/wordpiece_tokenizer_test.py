"""The test file for wordpiece tokenizer"""

import ehrudite.core.text as er_text
import ehrudite.core.tokenizer.wordpiece_tokenizer as wordpiece
import tempfile


MOCKED_EHRPREPER = "test-data/ehrprepered.xml"
VOCABULARY_SIZE = 128


def test_tokenizer_detokenizer():
    with tempfile.TemporaryDirectory() as dir_name:
        vocab_file = f"{dir_name}/vocab.txt"
        # Increase the number of Ehrpreper files to reach the minimun threshold of wordpiece,
        # otherwise there would be only single letters in the vocabulary.
        wordpiece.generate_vocab(
            [MOCKED_EHRPREPER for _ in range(3)], vocab_file, VOCABULARY_SIZE
        )
        tokenizer = wordpiece.WordpieceTokenizer(vocab_file)

        assert (
            VOCABULARY_SIZE * 0.5 <= tokenizer.vocab_size().numpy() <= VOCABULARY_SIZE
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

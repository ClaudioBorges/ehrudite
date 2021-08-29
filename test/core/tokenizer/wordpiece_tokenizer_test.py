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

        sentences = [
            "To be, or not to be, that is the question.",
            "Wisely and slow; they stumble that run fast.",
            "Some Cupid kills with arrows, some with traps.",
        ]
        lower_sentences = er_text.texts_to_sentences(sentences)
        for sentence in lower_sentences:
            words = list(er_text.sentences_to_words([sentence]))
            tokens = tokenizer.tokenize(words)
            detokens = tokenizer.detokenize(tokens)
            expected_detokens = [[word.encode()] for word in sentence.split()]
            assert all(expected_detokens == detokens.numpy()) == True

"""The test file for sentencepiece tokenizer"""

import ehrudite.core.text as er_text
import ehrudite.core.tokenizer.sentencepiece_tokenizer as sentencepiece
import tempfile


MOCKED_EHRPREPER = "test-data/ehrprepered.xml"
VOCABULARY_SIZE = 128


def test_tokenizer_detokenizer():
    with tempfile.TemporaryDirectory() as dir_name:
        output_prefix = f"{dir_name}/m"
        sentencepiece.generate_vocab([MOCKED_EHRPREPER], output_prefix, VOCABULARY_SIZE)
        model_file = f"{output_prefix}.model"
        tokenizer = sentencepiece.SentencepieceTokenizer(model_file)

        sentences = [
            "To be, or not to be, that is the question.",
            "Wisely and slow; they stumble that run fast.",
            "Some Cupid kills with arrows, some with traps.",
        ]
        lower_sentences = er_text.texts_to_sentences(sentences)
        for sentence in lower_sentences:
            tokens = tokenizer.tokenize(sentence)
            detokens = tokenizer.detokenize(tokens)
            assert sentence.encode() == detokens.numpy()

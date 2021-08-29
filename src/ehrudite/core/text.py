"""ehrpreper SentencePieceTokenizer"""

import ehrpreper
import sentencepiece as spm


class Generator:
    def __init__(self, raw_iterator):
        self._raw_iterator = raw_iterator

    def __call__(self):
        return self._raw_iterator


def texts_to_sentences(texts, to_lower=True):
    return (
        sentence.lower() if to_lower else sentence
        for text in texts
        for sentence in text.splitlines()
    )


def sentences_to_words(sentences):
    return (word for sentence in sentences for word in sentence.split())

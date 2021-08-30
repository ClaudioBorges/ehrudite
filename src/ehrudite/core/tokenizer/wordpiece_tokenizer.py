"""ehrpreper WordPieceTokenizer"""

from ehrudite.core.text import Generator
from tensorflow_text.tools.wordpiece_vocab import (
    wordpiece_tokenizer_learner_lib as learner,
)
import ehrpreper
import ehrudite.core.text as er_text
import tensorflow as tf
import tensorflow_text as tf_text


RESERVED_TOKES = ["[PAD]", "[UNK]", "[START]", "[END]"]


class WordpieceTokenizer:
    def __init__(self, model_file_name):
        self._wp = tf_text.WordpieceTokenizer(model_file_name)

    def tokenize(self, sentence):
        words = list(er_text.sentences_to_words([sentence]))
        return self._wp.tokenize(words)

    def detokenize(self, tokenized_words):
        words = self._wp.detokenize(tokenized_words)
        sentence = tf.strings.reduce_join(words, axis=0, separator=" ")
        return sentence


def generate_vocab(ehrpreper_files, output_file_name, vocab_size=32000):
    def write_vocab_file(filepath, vocab):
        with open(filepath, "w") as f:
            for token in vocab:
                print(token, file=f)

    texts = ehrpreper.data_generator(*ehrpreper_files)
    sentences = er_text.texts_to_sentences(texts)
    words = er_text.sentences_to_words(sentences)

    dataset = tf.data.Dataset.from_generator(
        Generator(words), output_types=tf.string, output_shapes=()
    )
    opt_dataset = dataset.batch(1000).prefetch(2)
    word_counts = learner.count_words(opt_dataset)

    vocab = learner.learn(word_counts, vocab_size, RESERVED_TOKES)
    write_vocab_file(output_file_name, vocab)

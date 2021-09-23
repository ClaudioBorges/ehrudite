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


class WordpieceTokenizer(tf_text.WordpieceTokenizer):
    def __init__(self, model_file_name):
        super(WordpieceTokenizer, self).__init__(model_file_name)

    def tokenize(self, sentence):
        words = list(er_text.split_into_words([sentence]))
        word_tokens = super(WordpieceTokenizer, self).tokenize(words)
        return word_tokens.merge_dims(0, -1)

    def detokenize(self, tokens):
        word_tokens = tf.reshape(tokens, [1, -1])
        sequences = super(WordpieceTokenizer, self).detokenize(word_tokens)
        words = sequences.merge_dims(0, -1)
        sentence = tf.strings.reduce_join(words, axis=0, separator=" ")
        return sentence

    def vocab_size(self):
        # Tensorflow text 2.6.0 does not have a vocab_size method
        return self._vocab_lookup_table.size()


def generate_vocab(ehrpreper_files, output_file_name, vocab_size=32000):
    texts = ehrpreper.data_generator(*ehrpreper_files)
    generate_vocab_from_texts(texts, output_file_name, vocab_size)


def generate_vocab_from_texts(texts, output_file_name, vocab_size=32000):
    def write_vocab_file(filepath, vocab):
        with open(filepath, "w") as f:
            for token in vocab:
                print(token, file=f)

    preprocessed = er_text.preprocess(texts)
    words = er_text.split_into_words(preprocessed)

    dataset = tf.data.Dataset.from_generator(
        Generator(words), output_types=tf.string, output_shapes=()
    )
    opt_dataset = dataset.batch(1000).prefetch(2)
    word_counts = learner.count_words(opt_dataset)

    vocab = learner.learn(word_counts, vocab_size, RESERVED_TOKES)
    write_vocab_file(output_file_name, vocab)

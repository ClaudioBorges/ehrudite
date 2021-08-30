"""ehrpreper SentencepieceTokenizer"""

from tensorflow.python.platform import gfile
import ehrpreper
import ehrudite.core.text as er_text
import sentencepiece as spm
import tensorflow as tf
import tensorflow_text as tf_text


class SentencepieceTokenizer:
    def __init__(self, model_file_name):
        model = gfile.GFile((model_file_name), "rb").read()
        self._tok = tf_text.SentencepieceTokenizer(model=model, out_type=tf.string)

    def tokenize(self, sentence):
        return self._tok.tokenize(sentence)

    def detokenize(self, tokenized_sentence):
        return self._tok.detokenize(tokenized_sentence)


def generate_vocab(ehrpreper_files, output_file_name_prefix, vocab_size=32000):
    texts = ehrpreper.data_generator(*ehrpreper_files)
    sentences = er_text.texts_to_sentences(texts)

    spm.SentencePieceTrainer.train(
        sentence_iterator=sentences,
        model_prefix=output_file_name_prefix,
        vocab_size=vocab_size,
    )

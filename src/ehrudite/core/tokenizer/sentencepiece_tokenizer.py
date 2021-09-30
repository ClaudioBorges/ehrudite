"""ehrpreper SentencepieceTokenizer"""

from tensorflow.python.platform import gfile
import ehrpreper
import ehrudite.core.text as er_text
import sentencepiece as spm
import tensorflow as tf
import tensorflow_text as tf_text

# To keep ehrudite tokenizer consistent, the special symbols are:
#  --------------------------------------------
# | Text  | Token Value | Description          |
# |-------|-------------|----------------------|
# | <pad> |      0      | Padding              |
# | <s>   |      1      | Begining of sentence |
# | </s>  |      2      | End of sentence      |
# | <unk> |      3      | Unknwon              |
#  --------------------------------------------


class SentencepieceTokenizer(tf_text.SentencepieceTokenizer):
    def __init__(self, model_file_name):
        model = gfile.GFile((f"{model_file_name}.model"), "rb").read()
        super(SentencepieceTokenizer, self).__init__(model=model)


def generate_vocab(ehrpreper_files, output_file_name, vocab_size=32000):
    texts = ehrpreper.data_generator(*ehrpreper_files)
    generate_vocab_from_texts(texts, output_file_name, vocab_size)


def generate_vocab_from_texts(texts, output_file_name, vocab_size=32000):
    preprocessed = er_text.preprocess(texts)

    spm.SentencePieceTrainer.train(
        sentence_iterator=preprocessed,
        model_prefix=output_file_name,
        vocab_size=vocab_size,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        max_sentence_length=2 ** 18,
    )

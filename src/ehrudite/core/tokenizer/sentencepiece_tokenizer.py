"""ehrpreper SentencePieceTokenizer"""

import ehrpreper
import sentencepiece as spm


class SentencePieceTokenizer:
    def __init__(self, model_file_name):
        self._sp = spm.SentencePieceProcessor(model_file=model_file_name)

    def tokenize(self, sentences):
        return self._sp.encode(sentences, out_type=str)

    def detokenize(self, sequences):
        return self._sp.decode(sequences)

    @staticmethod
    def generate_vocab(ehrpreper_files, output_file_name_prefix, vocab_size=32000):
        def make_sentences(ehrpreper_files):
            return (
                sentence
                for text in ehrpreper.data_generator(*ehrpreper_files)
                for sentence in text.splitlines()
            )

        spm.SentencePieceTrainer.train(
            sentence_iterator=make_sentences(ehrpreper_files),
            model_prefix=output_file_name_prefix,
            vocab_size=vocab_size,
        )


# input_file_name = "data.xml"
# output_file_name_prefix = "spm_d"
# SentencePieceTokenizer.generate_vocab([input_file_name], output_file_name_prefix)
# sentence = "The patient was sick with cought - Claudio Aparecido Borges Junior"
# tok = SentencePieceTokenizer(f'{output_file_name_prefix}.model')
# sentence_tokenize = tok.tokenize(sentence)
# sentence_detokenize = tok.detokenize(sentence_tokenize)
# print(sentence)
# print(sentence_tokenize)

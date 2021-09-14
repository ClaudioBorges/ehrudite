"""ehrudite pipeline"""


import ehrpreper
import ehrudite.core.text as er_text
from ehrudite.core.embedding.skipgram import SkipgramModel
from ehrudite.core.embedding.glove import GloVeModel


class EhruditePipeline:
    def __init__(self, ehrpreper_file, tokenizer):
        self._ehrpreper_file = ehrpreper_file
        self._tokenizer = tokenizer

    def _make_generator(self):
        sentences = self._make_contents()
        sequences = (
            self._tokenizer.tokenize(sentence).numpy().tolist()
            for sentence in sentences
        )
        for i, sequence in enumerate(sequences):
            yield sequence
            if i == 100:
                break

    def _tf_run(self):
        repeatable_gen = er_text.ProgressableGenerator(self._make_generator)
        import pdb

        pdb.set_trace()

        # model = GloVeModel(embedding_size=300, context_size=10)
        # model.fit_to_corpus(repeatable_gen)
        # model.train(num_epochs=100)

        skipgram = SkipgramModel()
        skipgram.fit_to_corpus(repeatable_gen)
        skipgram.train(epochs=20, window=100, workers=16)
        lll = list(model.wv.key_to_index.keys())

        # model.save("word2vec.model")

        # embedding = tf_skipgram.SkipGramModel(
        #    vocab_size=tokenizer.vocab_size().numpy(),
        #    embedding_dim=128,
        #    n_negative_samples=3,
        #    window_size=4,
        # )

        # sentence_ds = self._make_content_sentence_ds(self._ehrpreper_file)
        # sentence_ds = sentence_ds.cache().batch(1024).prefetch(tf.data.AUTOTUNE)
        # sequence_ds = sentence_ds.map(self._tokenizer.tokenize).unbatch()
        # embedding.run(sequence_ds.take(1000))
        # embedding.save("./skipgram.model")

    def run(self):
        self._tf_run()

        # SentencePiece
        # Word2Vec
        pass

    def _make_contents(self):
        texts = ehrpreper.data_generator(self._ehrpreper_file)
        preprocessed = er_text.preprocess(texts)
        return preprocessed

    def _make_content_sentence_ds(self, ehrpreper_file):
        sentences = self._make_contents()
        return tf.data.Dataset.from_generator(
            er_text.Generator(sentences), output_types=tf.string, output_shapes=()
        )
        # def _make_ehr_dataset(ehrpreper_file):
        #    def model_iterator():
        #        return (
        #            (document.content, document.annotations)
        #            for model in ehrpreper.load(ehrpreper_file)
        #            for document in model.documents
        #        )

        #    return tf.data.Dataset.from_generator(
        #        er_text.Generator(model_iterator()),
        #        output_signature=(
        #            tf.TensorSpec(shape=(), dtype=tf.string),
        #            tf.TensorSpec(shape=None, dtype=tf.string),
        #        ),
        #    )

        # ehr_ds = self._make_ehr_dataset(ehrpreper_file)

        ## [TODO] split the same as python string splitlines
        # sentence_ds = (
        #    ehr_ds.map(lambda content, annotations: content)
        #    .map(lambda content: tf.strings.lower(content))
        #    .flat_map(lambda x: tf.data.Dataset.from_tensor_slices(tf.strings.split(x, '\n')))
        #    .flat_map(lambda x: tf.data.Dataset.from_tensor_slices(tf.strings.split(x, '\r')))
        #    .flat_map(lambda x: tf.data.Dataset.from_tensor_slices(tf.strings.split(x, '\v')))
        #    .flat_map(lambda x: tf.data.Dataset.from_tensor_slices(tf.strings.split(x, '\f')))
        #    .flat_map(lambda x: tf.data.Dataset.from_tensor_slices(tf.strings.split(x, '\x1c')))
        #    .flat_map(lambda x: tf.data.Dataset.from_tensor_slices(tf.strings.split(x, '\x1d')))
        #    .flat_map(lambda x: tf.data.Dataset.from_tensor_slices(tf.strings.split(x, '\x1e')))
        #    .flat_map(lambda x: tf.data.Dataset.from_tensor_slices(tf.strings.split(x, '\x85')))
        #    .flat_map(lambda x: tf.data.Dataset.from_tensor_slices(tf.strings.split(x, '\u2028')))
        #    .flat_map(lambda x: tf.data.Dataset.from_tensor_slices(tf.strings.split(x, '\u2029')))
        #    .filter(lambda sentence: tf.strings.length(sentence) != 0)
        # )
        # return sentence_ds

    def _load_contents_from_ehrpreper(ehrpreper_file_name):
        logging.info(f"_load_contents_from_ehrpreper (file_name={ehrpreper_file_name})")
        return (
            document.content
            for model in ehrpreper.load(ehrpreper_file_name)
            for document in model.documents
        )


# [TODO] debug
if __name__ == "__main__":
    import ehrudite.core.tokenizer.sentencepiece_tokenizer as sentencepiece
    import ehrudite.core.tokenizer.wordpiece_tokenizer as wordpiece

    import tensorflow as tf

    ehrpreper_file = "/Users/clborges/Mestrado/code/data/ehrpreper.xml"
    tokenizer = sentencepiece.SentencepieceTokenizer(
        "/Users/clborges/Mestrado/code/data/tok/sentencepiece/sentencepiece.model"
    )
    # tokenizer = wordpiece.WordpieceTokenizer(
    #    "/Users/clborges/Mestrado/code/data/tok/wordpiece/wordpiece.vocab"
    # )

    ehr = EhruditePipeline(ehrpreper_file, tokenizer)
    ehr.run()

"""ehrudite pipeline"""


from ehrudite.core.embedding.glove import GloveModel
from ehrudite.core.embedding.skipgram import SkipgramModel
import ehrpreper
import ehrudite.core.text as er_text
import ehrudite.core.tokenizer.sentencepiece_tokenizer as sentencepiece
import ehrudite.core.tokenizer.wordpiece_tokenizer as wordpiece
import ipdb as pdb
import logging
import sklearn as skl
import sklearn.model_selection as skl_msel
import tensorflow as tf
import tqdm


class EhruditePipeline:
    def __init__(self, ehrpreper_file, tokenizer):
        self._ehrpreper_file = ehrpreper_file
        self._tokenizer = tokenizer

    def _make_vocabulary(self, tokenizer, output_path, **kwargs):
        tokenizer.generate_vocab_file_from_texts(texts, output_path, **kwargs)

    def _split_train_test(self, documents):
        pass

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
        repeatable_gen = er_text.RepeatableGenerator(self._make_generator)
        import pdb

        pdb.set_trace()

        # model = GloveModel(embedding_size=300, context_size=10)
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

    def _document_annotations_join(self, annotations, separator=" "):
        return separator.join(annotations)

    def _document_to_xy(self, documents):
        return (
            (
                er_text.preprocess_text(document.content),
                er_text.preprocess_text(
                    self._document_annotations_join(document.annotations)
                ),
            )
            for i, document in enumerate(documents)
        )

    def _data_xy_gen(self):
        func = lambda: self._document_to_xy(
            ehrpreper.document_entity_generator(self._ehrpreper_file)
        )
        return er_text.LenghtableRepeatableGenerator(func)

    def _data_xy_k_fold_gen(self, n_splits=4):
        data = self._data_xy_gen()
        logging.info(f"_k_fold_gen (n_splits={n_splits}, data_len={len(data)})")
        kf = skl_msel.KFold(n_splits=n_splits)
        # Make indexes to be sure data can be a generator
        source_idxs = [i for i in range(len(data))]
        for i, (
            train_idxs,
            test_idxs,
        ) in enumerate(kf.split(source_idxs)):
            logging.debug(
                f"Generating k-fold"
                + f"(index={i}, len(train)={len(train_idxs)}, len(test)={len(test_idxs)}"
            )

            def _k_fold_data(data, idxs):
                return (elm for idx, elm in enumerate(data) if idx in idxs)

            train = er_text.LenghtableRepeatableGenerator(
                _k_fold_data, _length=len(train_idxs), data=data, idxs=set(train_idxs)
            )
            test = er_text.LenghtableRepeatableGenerator(
                _k_fold_data, _length=len(test_idxs), data=data, idxs=set(test_idxs)
            )

            yield (
                train,
                test,
            )

    def _make_document_k_fold_xy_ds(self):
        generator = self._data_xy_k_fold_gen()
        return tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.string),
            ),
        )

    def _document_to_tuple(self, documents):
        return (
            (
                document.content,
                document.annotations,
            )
            for document in documents
        )

    def _make_document_ds(self):
        func = lambda: self._document_to_tuple(
            ehrpreper.document_entity_generator(self._ehrpreper_file)
        )
        generator = er_text.RepeatableGenerator(func)
        return tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(None,), dtype=tf.string),
            ),
        )

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


def _unpack_data_xy(data):
    def _extract_pos(pos):
        return (elm[pos] for elm in data)

    return (
        er_text.LenghtableRepeatableGenerator(_extract_pos, _length=len(data), pos=0),
        er_text.LenghtableRepeatableGenerator(_extract_pos, _length=len(data), pos=1),
    )


def run(run_id, train_xy, test_xy):
    train_x, train_y = _unpack_data_xy(train_xy)

    pdb.set_trace()

    # Sentencepiece Tokenize
    sentencepiece.generate_vocab_from_texts(
        (elm for elm in tqdm.tqdm(iterable=train_x)),
        f"train_x_{run_id}",
        vocab_size=8192,
    )
    sentencepiece.generate_vocab_from_texts(
        (elm for elm in tqdm.tqdm(iterable=train_y)),
        f"train_y_{run_id}",
        vocab_size=256,
    )
    # Wordpiece Tokenize
    wordpiece.generate_vocab_from_texts(
        (elm for elm in tqdm.tqdm(iterable=train_x)),
        f"train_x_{run_id}",
        vocab_size=8192,
    )
    wordpiece.generate_vocab_from_texts(
        (elm for elm in tqdm.tqdm(iterable=train_y)),
        f"train_y_{run_id}",
        vocab_size=256,
    )

    # Here
    pass


# [TODO] debug
if __name__ == "__main__":
    import ehrudite.cli.base as cli_base

    cli_base.config_logging(3)

    ehrpreper_file = "/Users/clborges/Mestrado/code/data/ehrpreper.xml"
    # tokenizer = sentencepiece.SentencepieceTokenizer(
    #    "/Users/clborges/Mestrado/code/data/tok/sentencepiece/sentencepiece.model"
    # )
    # tokenizer = wordpiece.WordpieceTokenizer(
    #    "/Users/clborges/Mestrado/code/data/tok/wordpiece/wordpiece.vocab"
    # )

    ehr = EhruditePipeline(ehrpreper_file, None)

    for run_id, (
        train_xy,
        test_xy,
    ) in enumerate(ehr._data_xy_k_fold_gen()):
        run(run_id, train_xy, test_xy)
        break

    print("End")

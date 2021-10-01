"""ehrudite pipeline"""


from ehrudite.core.embedding.glove import GloveModel
from ehrudite.core.embedding.skipgram import SkipgramModel
import ehrpreper
import ehrudite.core.dnn.transformer as transformer_m
import ehrudite.core.text as er_text
import ehrudite.core.tokenizer.sentencepiece_tokenizer as sentencepiece
import ehrudite.core.tokenizer.wordpiece_tokenizer as wordpiece
import ipdb as pdb
import logging
import os
import sklearn.model_selection as skl_msel
import tensorflow as tf
import time
import tqdm

ehr_data_path = "../ehr-data/"
tokenizer_base_path = os.path.join(ehr_data_path, "tokenizer/")
sentencepiece_base_path = os.path.join(tokenizer_base_path, "sentencepiece/")
wordpiece_base_path = os.path.join(tokenizer_base_path, "wordpiece/")
checkpoint_path = os.path.join(ehr_data_path, "checkpoint/train")

PAD_TOK = 0
BOS_TOK = 1
EOS_TOK = 2
UNK_TOK = 3


class EhruditePipeline:
    def __init__(self, ehrpreper_file, tokenizer):
        self._ehrpreper_file = ehrpreper_file
        self._tokenizer = tokenizer

    def run(self):
        self._tf_run()

        # SentencePiece
        # Word2Vec
        pass

    def _join_annotations(self, annotations, separator=" "):
        return separator.join(annotations)

    def _prepare_and_split_x_y(self, documents):
        return (
            (
                er_text.preprocess_text(document.content),
                er_text.preprocess_text(self._join_annotations(document.annotations)),
            )
            for i, document in enumerate(documents)
        )

    def _ehrpreper_gen(self):
        func = lambda: self._prepare_and_split_x_y(
            ehrpreper.document_entity_generator(self._ehrpreper_file)
        )
        return er_text.LenghtableRepeatableGenerator(func)

    def _ehrpreper_k_fold_gen(self, n_splits=4):
        data = self._ehrpreper_gen()
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


def _enclose_tensor_with_bos_eos(tensor):
    tf.concat([[BOS_TOK], tensor, [EOS_TOK]], 0)


def _unpack_data_xy(data):
    def _extract_pos(pos):
        return (elm[pos] for elm in data)

    return (
        er_text.LenghtableRepeatableGenerator(_extract_pos, _length=len(data), pos=0),
        er_text.LenghtableRepeatableGenerator(_extract_pos, _length=len(data), pos=1),
    )


def _wrap_with_tqdm(iterable):
    return (i for i in tqdm.tqdm(iterable=iterable))


def _train_tokenizer(run_id, train_xy, test_xy):
    train_x, train_y = _unpack_data_xy(train_xy)

    vocab_size_x = 2 ** 14
    vocab_size_y = 2 ** 9
    train_x_id = f"train_x_{run_id}"
    train_y_id = f"train_y_{run_id}"

    # Sentencepiece Tokenize
    sentencepiece_x_y = (
        os.path.join(sentencepiece_base_path, train_x_id),
        os.path.join(sentencepiece_base_path, train_y_id),
    )
    sentencepiece.generate_vocab_from_texts(
        _wrap_with_tqdm(train_x), sentencepiece_x_y[0], vocab_size=vocab_size_x
    )
    sentencepiece.generate_vocab_from_texts(
        _wrap_with_tqdm(train_y), sentencepiece_x_y[1], vocab_size=vocab_size_y
    )

    # Wordpiece Tokenize
    wordpiece_x_y = (
        os.path.join(wordpiece_base_path, train_x_id),
        os.path.join(wordpiece_base_path, train_y_id),
    )
    wordpiece.generate_vocab_from_texts(
        _wrap_with_tqdm(train_x), wordpiece_x_y[0], vocab_size=vocab_size_x
    )
    wordpiece.generate_vocab_from_texts(
        _wrap_with_tqdm(train_y), wordpiece_x_y[1], vocab_size=vocab_size_y
    )


def _train_model(run_id, train_xy, test_xy):
    train_x, train_y = _unpack_data_xy(train_xy)

    train_x_id = f"train_x_{run_id}"
    train_y_id = f"train_y_{run_id}"

    # [TODO] Use wordpiece
    tok_input = wordpiece.WordpieceTokenizer(
        os.path.join(wordpiece_base_path, train_x_id)
    )
    tok_target = wordpiece.WordpieceTokenizer(
        os.path.join(wordpiece_base_path, train_y_id)
    )

    tok_s = sentencepiece.SentencepieceTokenizer(
        os.path.join(sentencepiece_base_path, train_x_id)
    )

    X_MAX_LEN = 1024
    Y_MAX_LEN = 128

    def _prepare_gen():
        def normalize(sequence, maxlen):
            sliced = tf.slice(sequence, [0], [min(maxlen - 2, sequence.shape[0])])
            enclosed = tf.concat([[BOS_TOK], sliced, [EOS_TOK]], 0)
            padded = tf.keras.preprocessing.sequence.pad_sequences(
                [enclosed],
                maxlen=maxlen,
                padding="post",
                truncating="post",
                value=PAD_TOK,
            )[0]
            return tf.constant(padded)

        train_x_tok = (tok_input.tokenize(x) for x in train_x)
        train_y_tok = (tok_target.tokenize(y) for y in train_y)

        return (
            (normalize(x, X_MAX_LEN), normalize(y, Y_MAX_LEN))
            for x, y in zip(train_x_tok, train_y_tok)
        )

    train_xy_tok_gen = er_text.RepeatableGenerator(_prepare_gen)

    train_xy_tok_ds = tf.data.Dataset.from_generator(
        train_xy_tok_gen,
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
        ),
    )

    BUFFER_SIZE = 128  # 20000
    BATCH_SIZE = 64

    def make_batches(ds):
        return (
            ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        )

    train_batches = make_batches(train_xy_tok_ds)

    # From https://www.tensorflow.org/text/tutorials/transformer
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    def accuracy_function(real, pred):
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")

    transformer = transformer_m.Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=tok_input.vocab_size(),
        target_vocab_size=tok_target.vocab_size(),
        pe_input=1000,
        pe_target=1000,
        rate=dropout_rate,
    )

    optimizer = transformer_m.optimizer(d_model)

    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!!")

    EPOCHS = 20
    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = transformer([inp, tar_inp], training=True)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_batches):
            train_step(inp, tar)

            if batch % 50 == 0:
                print(
                    f"Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}"
                )

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f"Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}")

        print(
            f"Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}"
        )

        print(f"Time taken for 1 epoch: {time.time() - start:.2f} secs\n")


# [TODO] debug
if __name__ == "__main__":
    import ehrudite.cli.base as cli_base

    cli_base.config_logging(3)

    ehrpreper_file = os.path.join(ehr_data_path, "ehrpreper.xml")

    ehr = EhruditePipeline(ehrpreper_file, None)
    for run_id, (
        train_xy,
        test_xy,
    ) in enumerate(ehr._ehrpreper_k_fold_gen()):
        # _train_tokenizer(run_id, train_xy, test_xy)
        _train_model(run_id, train_xy, test_xy)

    print("End")

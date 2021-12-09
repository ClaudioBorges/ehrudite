"""ehrudite tokenizer pipeline"""

import Levenshtein
import ehrudite.core.dnn.lstm as bi_lstm_attn_m
import ehrudite.core.dnn.transformer as transformer_m
import ehrudite.core.pipeline as pip
import ehrudite.core.pipeline.tokenizer as pip_tok
import ehrudite.core.text as er_text
import enum
import logging
import os
import pathlib as pathlib
import sklearn
import tensorflow as tf
import time


X_MAX_LEN = 512
Y_MAX_LEN = 64
BUFFER_SIZE = 1024
#BATCH_SIZE = 144
BATCH_SIZE = 245
#BATCH_SIZE = 34
ACCURACY_TH = 0.80
# Used for testing a subset of the entire corpus. Use -1 to train with full corpus
CORPUS_LIMIT = -1
MODEL_CHECKPOINT_BASE_PATH = os.path.join(pip.BASE_PATH, "model/checkpoint/train/")


# From https://www.tensorflow.org/text/tutorials/transformer
XFMR_XFMR_NUM_LAYERS = 2  # 3 # 4  # 8  # 4 # 6
XFMR_XFMR_D_MODEL = 512
XFMR_XFMR_DFF = 2048
XFMR_XFMR_NUM_HEADS = 8
XMFR_XFMR_DROPOUT_RATE = 0.1


# LSTM_LSTM
LSTM_LSTM_EMBEDDING_DIM_INP = 512
LSTM_LSTM_EMBEDDING_DIM_TAR = 32
LSTM_LSTM_RNN_UNITS = 128


class DnnType(enum.Enum):
    XFMR_XFMR = 1
    XFMR_LSTM = 2
    LSTM_XFMR = 3
    LSTM_LSTM = 4


SPECIFICATIONS = {
    "base_ckpt": MODEL_CHECKPOINT_BASE_PATH,
}


def sequence_enclose(sequence, max_length):
    sliced = tf.slice(sequence, [0], [min(max_length - 2, sequence.shape[0])])
    return tf.concat([[pip_tok.BOS_TOK], sliced, [pip_tok.EOS_TOK]], 0)


def sequence_pad(sequence, max_length, padding="post", truncating="post"):
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        [sequence],
        maxlen=max_length,
        padding=padding,
        truncating=truncating,
        value=pip_tok.PAD_TOK,
    )[0]
    return tf.constant(padded)


def sequence_pad_and_enclose(sequence, max_length, **kwargs):
    sequence = sequence_enclose(sequence, max_length, **kwargs)
    return sequence_pad(sequence, max_length, **kwargs)


def restore_or_init(run_id, dnn_type, tokenizer_type, restore=True):
    logging.info(
        f"Restore or init (run_id={run_id}, dnn={dnn_type}, tok={tokenizer_type})"
    )
    tok_x, tok_y = pip_tok.restore(run_id, tokenizer_type)
    base_path = SPECIFICATIONS["base_ckpt"]

    if dnn_type == DnnType.XFMR_XFMR:
        model = transformer_m.Transformer(
            num_layers=XFMR_XFMR_NUM_LAYERS,
            d_model=XFMR_XFMR_D_MODEL,
            num_heads=XFMR_XFMR_NUM_HEADS,
            dff=XFMR_XFMR_DFF,
            input_vocab_size=tok_x.vocab_size(),
            target_vocab_size=tok_y.vocab_size(),
            pe_input=X_MAX_LEN,
            pe_target=Y_MAX_LEN,
            rate=XMFR_XFMR_DROPOUT_RATE,
        )
        optimizer = transformer_m.optimizer(XFMR_XFMR_D_MODEL)
        ckpt_path = os.path.join(
            base_path,
            str(dnn_type),
            str(tokenizer_type),
            f"run_id={run_id}",
            f"corpus_limit={CORPUS_LIMIT}",
            f"num_layers={XFMR_XFMR_NUM_LAYERS}",
            f"num_heads={XFMR_XFMR_NUM_HEADS}",
            f"d_model={XFMR_XFMR_D_MODEL}",
            f"dff={XFMR_XFMR_DFF}",
            f"x_len={X_MAX_LEN}",
            f"y_max_len={Y_MAX_LEN}",
            f"dropout_rate={XMFR_XFMR_DROPOUT_RATE}",
        )

    elif dnn_type == DnnType.LSTM_LSTM:
        model = bi_lstm_attn_m.Seq2SeqBiLstmAttn(
            embedding_dim_inp=LSTM_LSTM_EMBEDDING_DIM_INP,
            embedding_dim_tar=LSTM_LSTM_EMBEDDING_DIM_TAR,
            units=LSTM_LSTM_RNN_UNITS,
            input_vocab_size=tok_x.vocab_size(),
            target_vocab_size=tok_y.vocab_size(),
        )
        optimizer = tf.optimizers.Adam()
        ckpt_path = os.path.join(
            base_path,
            str(dnn_type),
            str(tokenizer_type),
            f"run_id={run_id}",
            f"corpus_limit={CORPUS_LIMIT}",
            f"embedding_dim_inp={LSTM_LSTM_EMBEDDING_DIM_INP}",
            f"embedding_dim_tar={LSTM_LSTM_EMBEDDING_DIM_TAR}",
            f"units={LSTM_LSTM_RNN_UNITS}",
            f"x_len={X_MAX_LEN}",
            f"y_max_len={Y_MAX_LEN}",
        )

    # Create the path if it doesn't exist
    pathlib.Path(ckpt_path).mkdir(parents=True, exist_ok=True)
    logging.info(f"Pipeline path (path={ckpt_path})")

    ckpt = tf.train.Checkpoint(transformer=model, optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=5)
    if restore:
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            logging.info(f"Latest checkpoint restored (path={ckpt_path})")
        else:
            logging.info(f"No checkpoing found, model initialized (path={ckpt_path})")
    else:
        logging.info(f"Restore disabled, model initialized (path={ckpt_path})")

    return (
        model,
        optimizer,
        ckpt_manager,
    )


class Translator(tf.Module):
    def __init__(self, run_id, dnn_type, tokenizer_type, model=None):
        self.tok_x, self.tok_y = pip_tok.restore(run_id, tokenizer_type)
        self.model = model
        if self.model is None:
            self.model, _, _ = restore_or_init(run_id, dnn_type, tokenizer_type)

    def __call__(self, sentence):
        # Input sentence is the EHR, hence preparing with BOS and EOS
        sequence = self.tok_x.tokenize(sentence)
        encoder_input = sequence_pad_and_enclose(sequence, X_MAX_LEN)[tf.newaxis]

        bos_token = tf.constant(pip_tok.BOS_TOK, dtype=tf.int64)[tf.newaxis]
        eos_token = tf.constant(pip_tok.EOS_TOK, dtype=tf.int64)[tf.newaxis]

        # `tf.TensorArray` is required here (instead of a python list) so that the
        # dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, bos_token)

        for i in tf.range(Y_MAX_LEN - 1):
            output = tf.transpose(output_array.stack())
            predictions = self.model([encoder_input, output], training=False)[0]

            # select the last token from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.argmax(predictions, axis=-1)

            # concatenate the predicted_id to the output which is given to the decoder
            # as its input
            output_array = output_array.write(i + 1, predicted_id[0])

            if predicted_id == eos_token:
                break

        output = tf.transpose(output_array.stack())
        text = self.tok_y.detokenize(tf.cast(output, dtype=tf.int32))[0]  # shape: ()

        tokens = output  # self.tok_y.lookup(tf.cast(output, dtype=tf.int32))[0]

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop. So recalculate them outside
        # the loop.
        attention_weights = self.model([encoder_input, output[:, :-1]], training=False)[
            1
        ]

        return text, tokens, attention_weights


def validate(run_id, dnn_type, tokenizer_type, train_xy, test_xy):
    translator = Translator(run_id, dnn_type, tokenizer_type)
    _, tok_y = pip_tok.restore(run_id, tokenizer_type)

    def accuracy(real, pred):
        accuracies = tf.equal(real, pred)

        mask = tf.math.logical_not(tf.math.equal(real, pip_tok.PAD_TOK))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    def edit_distance(real, pred):
        # Make it irrelevant to the code order
        hypothesis_words = sorted(pred.split(" "))
        truth_words = sorted(real.split(" "))

        hyphothesis = " ".join(hypothesis_words)
        truth = " ".join(truth_words)

        distance = Levenshtein.distance(hyphothesis, truth)
        # Normalize the distance
        return distance / max(len(hyphothesis), len(truth))

    def f1_score(real, pred, average):
        # Make it irrelevant to the order
        pred = sorted(pred.split(" "))
        real = sorted(real.split(" "))

        max_length = max(len(pred), len(real))

        # Add padding if necessary
        pred = pred + [" "] * (max_length - len(pred))
        real = real + [" "] * (max_length - len(real))

        return sklearn.metrics.f1_score(real, pred, average=average)

    val_tok_accuracy = tf.keras.metrics.Mean(name="val_tok_accuracy")
    val_edit_distance = tf.keras.metrics.Mean(name="val_edit_distance")
    val_f1_score_micro = tf.keras.metrics.Mean(name="val_f1_score_micro")
    val_f1_score_macro = tf.keras.metrics.Mean(name="val_f1_score_macro")
    logging.info(f"Validating model...")

    start = time.time()
    for (i, (x, y)) in enumerate(test_xy):
        pred_text, pred_tokens, attention_weights = translator(x)

        pred_text = pred_text.numpy().decode()
        val_edit_distance(edit_distance(y, pred_text))
        val_f1_score_micro(f1_score(y, pred_text, "micro"))
        val_f1_score_macro(f1_score(y, pred_text, "macro"))

        pred_tokens = sequence_pad(pred_tokens[0], Y_MAX_LEN)
        real_tokens = sequence_pad_and_enclose(tok_y.tokenize(y), Y_MAX_LEN)
        val_tok_accuracy(accuracy(real_tokens, pred_tokens))

        if i > 0 and i % 1 == 0:
            time_diff = time.time() - start
            logging.info(
                f"Partial result (itetarion={i}, "
                f"tok_accuracy={val_tok_accuracy.result():.4f}, "
                f"edit_distance={val_edit_distance.result():.4f}, "
                f"f1_score_micro={val_f1_score_micro.result():.4f}, "
                f"f1_score_macro={val_f1_score_macro.result():.4f}, "
                f"time_diff={time_diff:.2f}s)"
            )
            logging.info(
                f"Partial result content (real_text={y}, pred_text={pred_text}, "
                f"real_tokens={real_tokens}, pred_tokens={pred_tokens})"
            )

    time_diff = time.time() - start
    logging.info(
        f"Validation finished ("
        f"tok_accuracy={val_tok_accuracy.result():.4f}, "
        f"edit_distance={val_edit_distance.result():.4f}, "
        f"f1_score_micro={val_f1_score_micro.result():.4f}, "
        f"f1_score_macro={val_f1_score_macro.result():.4f}, "
        f"time_diff={time_diff:.2f}s"
    )


def train(run_id, dnn_type, tokenizer_type, train_xy, use_checkpoint=True, **kwargs):
    # All models have an process, optimizer, a checkpoint manager and tokens
    xy_tokens = prepare_xy_tokens(run_id, tokenizer_type, train_xy)
    model, optimizer, ckpt_manager = restore_or_init(
        run_id, dnn_type, tokenizer_type, restore=use_checkpoint
    )

    if dnn_type == DnnType.XFMR_XFMR:
        train_xfmr_xfmr(run_id, model, optimizer, ckpt_manager, xy_tokens, **kwargs)
    elif dnn_type == DnnType.LSTM_LSTM:
        train_lstm_lstm(run_id, model, optimizer, ckpt_manager, xy_tokens, **kwargs)


def prepare_xy_tokens(run_id, tokenizer_type, xy):
    logging.info(
        "{prepare_xy_tokens.__name__} (run_id={run_id}, tokenizer_type={tokenizer_type})"
    )
    x, y = pip.unpack_2d(xy)
    tok_x, tok_y = pip_tok.restore(run_id, tokenizer_type)

    # DEBUG - Fixed elements
    if CORPUS_LIMIT is not None and CORPUS_LIMIT != -1:
        logging.info(f"Corpus LIMITED (corpus_limit={CORPUS_LIMIT})")
        debug_x = []
        debug_y = []
        for i, (x, y) in enumerate(zip(x, y)):
            debug_x.append(x)
            debug_y.append(y)
            if i >= (CORPUS_LIMIT - 1):
                break
        x = debug_x
        y = debug_y

    return er_text.LenghtableRepeatableGenerator(
        lambda: ((tok_x.tokenize(x1), tok_y.tokenize(y1)) for x1, y1 in zip(x, y)),
        _length=len(xy),
    )


def normalize_xy_tokens(xy_tokens, **kwargs):
    logging.info(f"{normalize_xy_tokens.__name__}")
    return (
        (sequence_pad(x, X_MAX_LEN, **kwargs), sequence_pad(y, Y_MAX_LEN, **kwargs))
        for x, y in xy_tokens
    )


def sequence_enclose_xy_tokens(xy_tokens):
    logging.info(f"{sequence_enclose_xy_tokens.__name__}")
    return (
        (sequence_enclose(x, X_MAX_LEN), sequence_enclose(y, Y_MAX_LEN))
        for x, y in xy_tokens
    )


def batch_xy_tokens(generator):
    logging.info("Making batch...")
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
        ),
    )
    return (
        dataset.cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


def make_xy_tokens_combinations(xy_tokens):
    return (
        (x_tokens, y_tokens[: i + 1])
        for x_tokens, y_tokens in xy_tokens
        for i in range(1, len(y_tokens))
    )


def train_xfmr_xfmr(run_id, model, optimizer, ckpt_manager, xy_tokens, n_epochs=None):
    logging.info(
        f"Training DNN (run_id={run_id}, dnn={DnnType.XFMR_XFMR}, n_epochs={n_epochs}"
    )

    train_xy_tokens = sequence_enclose_xy_tokens(xy_tokens)
    train_xy_norms = normalize_xy_tokens(train_xy_tokens)
    train_batches = batch_xy_tokens(train_xy_norms)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")

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
            loss = loss_function(loss_object, tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))

    main_train_loop(
        train_batches,
        train_loss,
        train_accuracy,
        train_step,
        n_epochs=n_epochs,
        ckpt_manager=ckpt_manager,
    )


def loss_function(loss_object, real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, pip_tok.PAD_TOK))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, pip_tok.PAD_TOK))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


def train_lstm_lstm(run_id, model, optimizer, ckpt_manager, xy_tokens, n_epochs=None):
    logging.info(
        f"Training DNN (run_id={run_id}, dnn={DnnType.LSTM_LSTM}, n_epochs={n_epochs})"
    )

    train_xy_tokens = sequence_enclose_xy_tokens(xy_tokens)
    train_xy_combs = make_xy_tokens_combinations(train_xy_tokens)
    train_xy_norms = normalize_xy_tokens(train_xy_combs, padding="pre")
    train_batches = batch_xy_tokens(train_xy_norms)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")

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
            predictions, _, _, _ = model([inp, tar_inp], training=True)
            loss = loss_function(loss_object, tar_real, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))

    main_train_loop(
        train_batches,
        train_loss,
        train_accuracy,
        train_step,
        n_epochs=n_epochs,
        ckpt_manager=ckpt_manager,
    )


def main_train_loop(
    train_batches,
    train_loss,
    train_accuracy,
    fc_train_step,
    n_epochs=None,
    ckpt_manager=None,
):
    if n_epochs:
        logging.info(f"Training for n_epochs(n_epochs={n_epochs})")
    else:
        logging.info(f"Training until threshold(accuracy_th={ACCURACY_TH})")

    epoch = 0
    while 1:
        if n_epochs is None:
            if train_accuracy.result() >= ACCURACY_TH:
                logging.info(
                    f"Accuracy reached: Epoch {epoch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}"
                )
                break
        elif n_epochs == epoch:
            logging.info(
                f"Epoch reached: Epoch {epoch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}"
            )
            break

        epoch += 1

        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> ehr, tar -> icd classification
        for batch_id, (
            inp,
            tar,
        ) in enumerate(train_batches):
            fc_train_step(inp, tar)

            if batch_id > 0 and batch_id % 1 == 0:
                logging.info(
                    f"Epoch {epoch} Batch {batch_id} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}"
                )
                logging.info(f"Elapsed time: {time.time() - start:.2f} secs")

        if ckpt_manager is not None:
            ckpt_save_path = ckpt_manager.save()
            logging.info(f"Saving checkpoint for epoch {epoch} at {ckpt_save_path}")

        logging.info(
            f"Epoch {epoch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}"
        )

        logging.info(f"Time taken for 1 epoch: {time.time() - start:.2f} secs\n")

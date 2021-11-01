"""ehrudite tokenizer pipeline"""

import Levenshtein
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


X_MAX_LEN = 2048
Y_MAX_LEN = 128
BUFFER_SIZE = 128
BATCH_SIZE = 8
# From https://www.tensorflow.org/text/tutorials/transformer
NUM_LAYERS = 2  # 3 # 4  # 8  # 4 # 6
D_MODEL = 512
DFF = 2048
NUM_HEADS = 8
DROPOUT_RATE = 0.1
ACCURACY_TH = 0.90
# Used for testing a subset of the entire corpus. Use -1 to train with full corpus
CORPUS_LIMIT = -1

MODEL_CHECKPOINT_BASE_PATH = os.path.join(pip.BASE_PATH, "model/checkpoint/train/")


class DnnType(enum.Enum):
    XFMR_XFMR = 1
    XFMR_LSTM = 2
    LSTM_XFMR = 3
    LSTM_LSTM = 4


SPECIFICATIONS = {
    "base_ckpt": MODEL_CHECKPOINT_BASE_PATH,
}


def normalize(sequence, max_length, add_bos_eof=True):
    if add_bos_eof:
        sliced = tf.slice(sequence, [0], [min(max_length - 2, sequence.shape[0])])
        enclosed = tf.concat([[pip_tok.BOS_TOK], sliced, [pip_tok.EOS_TOK]], 0)
        sequence = enclosed

    padded = tf.keras.preprocessing.sequence.pad_sequences(
        [sequence],
        maxlen=max_length,
        padding="post",
        truncating="post",
        value=pip_tok.PAD_TOK,
    )[0]
    return tf.constant(padded)


def restore_or_init(run_id, dnn_type, tokenizer_type, restore=True):
    logging.info(
        f"Restore or init (run_id={run_id}, dnn={dnn_type}, tok={tokenizer_type})"
    )
    tok_x, tok_y = pip_tok.restore(run_id, tokenizer_type)

    transformer = transformer_m.Transformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        input_vocab_size=tok_x.vocab_size(),
        target_vocab_size=tok_y.vocab_size(),
        pe_input=X_MAX_LEN,
        pe_target=Y_MAX_LEN,
        rate=DROPOUT_RATE,
    )

    base_path = SPECIFICATIONS["base_ckpt"]
    ckpt_path = os.path.join(
        base_path,
        str(dnn_type),
        str(tokenizer_type),
        f"run_id={run_id}",
        f"corpus_limit={CORPUS_LIMIT}",
        f"num_layers={NUM_LAYERS}",
        f"num_heads={NUM_HEADS}",
        f"d_model={D_MODEL}",
        f"dff={DFF}",
        f"x_len={X_MAX_LEN}",
        f"y_max_len={Y_MAX_LEN}",
        f"dropout_rate={DROPOUT_RATE}",
    )
    # Create the path if it doesn't exist
    pathlib.Path(ckpt_path).mkdir(parents=True, exist_ok=True)
    logging.info(f"Pipeline path (path={ckpt_path})")

    optimizer = transformer_m.optimizer(D_MODEL)
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

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
        transformer,
        optimizer,
        ckpt_manager,
    )


class Translator(tf.Module):
    def __init__(self, run_id, dnn_type, tokenizer_type, transformer=None):
        self.tok_x, self.tok_y = pip_tok.restore(run_id, tokenizer_type)
        self.transformer = transformer
        if self.transformer is None:
            self.transformer, _, _ = restore_or_init(run_id, dnn_type, tokenizer_type)

    def __call__(self, sentence):
        # Input sentence is the EHR, hence preparing with BOS and EOS
        sequence = self.tok_x.tokenize(sentence)
        encoder_input = normalize(sequence, X_MAX_LEN)[tf.newaxis]

        bos_token = tf.constant(pip_tok.BOS_TOK, dtype=tf.int64)[tf.newaxis]
        eos_token = tf.constant(pip_tok.EOS_TOK, dtype=tf.int64)[tf.newaxis]

        # `tf.TensorArray` is required here (instead of a python list) so that the
        # dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, bos_token)

        for i in tf.range(Y_MAX_LEN - 1):
            output = tf.transpose(output_array.stack())
            predictions, _ = self.transformer([encoder_input, output], training=False)

            # select the last token from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.argmax(predictions, axis=-1)

            # concatenate the predicted_id to the output which is given to the decoder
            # as its input
            output_array = output_array.write(i + 1, predicted_id[0])

            if predicted_id == eos_token:
                break

        # import pdb
        # pdb.set_trace()
        output = tf.transpose(output_array.stack())
        # output.shape(1, tokens)
        text = self.tok_y.detokenize(tf.cast(output, dtype=tf.int32))[0]  # shape: ()

        tokens = output  # self.tok_y.lookup(tf.cast(output, dtype=tf.int32))[0]

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop. So recalculate them outside
        # the loop.
        _, attention_weights = self.transformer(
            [encoder_input, output[:, :-1]], training=False
        )

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
        # for (i, (x, y)) in enumerate(train_xy):
        pred_text, pred_tokens, attention_weights = translator(x)

        pred_text = pred_text.numpy().decode()
        val_edit_distance(edit_distance(y, pred_text))
        val_f1_score_micro(f1_score(y, pred_text, "micro"))
        val_f1_score_macro(f1_score(y, pred_text, "macro"))

        pred_tokens = normalize(pred_tokens[0], Y_MAX_LEN, add_bos_eof=False)
        real_tokens = normalize(tok_y.tokenize(y), Y_MAX_LEN, add_bos_eof=True)
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


def train(run_id, dnn_type, tokenizer_type, train_xy, test_xy, **kwargs):
    if dnn_type == DnnType.XFMR_XFMR:
        train_xfmr_xfmr(run_id, tokenizer_type, train_xy, test_xy, **kwargs)


def train_xfmr_xfmr(
    run_id, tokenizer_type, train_xy, test_xy, restore=True, save=True, n_epochs=None
):
    logging.info(
        f"Training DNN (run_id={run_id}, dnn={DnnType.XFMR_XFMR},"
        f"tok={tokenizer_type})"
    )
    train_x, train_y = pip.unpack_2d(train_xy)

    logging.info("Restoring tokenizer...")
    tok_x, tok_y = pip_tok.restore(run_id, tokenizer_type)

    # DEBUG - Fixed elements
    if CORPUS_LIMIT is not None and CORPUS_LIMIT != -1:
        debug_x = []
        debug_y = []
        for i, (x, y) in enumerate(zip(train_x, train_y)):
            debug_x.append(x)
            debug_y.append(y)
            if i >= (CORPUS_LIMIT - 1):
                break
        train_x = debug_x
        train_y = debug_y

    def prepare_xy():
        train_x_tok = (tok_x.tokenize(x) for x in train_x)
        train_y_tok = (tok_y.tokenize(y) for y in train_y)

        return (
            (normalize(x, X_MAX_LEN), normalize(y, Y_MAX_LEN))
            for x, y in zip(train_x_tok, train_y_tok)
        )

    logging.info("Preparing x and y...")
    train_xy_tok_gen = er_text.LenghtableRepeatableGenerator(
        prepare_xy, _length=len(train_xy)
    )

    logging.info("Creating dataset...")
    train_xy_tok_ds = tf.data.Dataset.from_generator(
        train_xy_tok_gen,
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
        ),
    )

    def make_batches(ds):
        return (
            ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        )

    train_batches = make_batches(train_xy_tok_ds)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )

    def loss_function(real, pred):
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

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")

    logging.info("Restoring dnn...")
    transformer, optimizer, ckpt_manager = restore_or_init(
        run_id, DnnType.XFMR_XFMR, tokenizer_type, restore=restore
    )

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

    logging.info(f"Training (accuracy={ACCURACY_TH})")
    epoch = 0
    while 1:
        epoch += 1
        if n_epochs is None:
            if train_accuracy.result() >= ACCURACY_TH:
                logging.info(
                    f"Accuracy reached: Epoch {epoch} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}"
                )
                break
        elif n_epochs == epoch - 1:
            logging.info(
                f"Epoch reached: Epoch {epoch} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}"
            )
            break

        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> ehr, tar -> icd classification
        for (batch, (inp, tar)) in enumerate(train_batches):
            train_step(inp, tar)

            if batch > 0 and batch % 50 == 0:
                logging.info(
                    f"Epoch {epoch} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}"
                )
                logging.info(f"Elapsed time: {time.time() - start:.2f} secs")

        if save:
            ckpt_save_path = ckpt_manager.save()
            logging.info(f"Saving checkpoint for epoch {epoch} at {ckpt_save_path}")

        logging.info(
            f"Epoch {epoch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}"
        )

        logging.info(f"Time taken for 1 epoch: {time.time() - start:.2f} secs\n")

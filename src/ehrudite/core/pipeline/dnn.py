"""ehrudite tokenizer pipeline"""

import ehrudite.core.dnn.transformer as transformer_m
import ehrudite.core.pipeline as pip
import ehrudite.core.pipeline.tokenizer as pip_tok
import ehrudite.core.text as er_text
import enum
import logging
import os
import pathlib as pathlib
import tensorflow as tf
import time


X_MAX_LEN = 2048
Y_MAX_LEN = 128
BUFFER_SIZE = 128
BATCH_SIZE = 8
# From https://www.tensorflow.org/text/tutorials/transformer
NUM_LAYERS = 3 # 4  # 8  # 4 # 6
D_MODEL = 512  # 128 # 512
DFF = 2048  # 512 # 2048
NUM_HEADS = 8
DROPOUT_RATE = 0.1
ACCURACY_TH = 0.8

MODEL_CHECKPOINT_BASE_PATH = os.path.join(pip.BASE_PATH, "model/checkpoint/train/")


class DnnType(enum.Enum):
    XFMR_XFMR = 1
    XFMR_LSTM = 2
    LSTM_XFMR = 3
    LSTM_LSTM = 4


SPECIFICATIONS = {
    "base_ckpt": MODEL_CHECKPOINT_BASE_PATH,
}


def normalize(sequence, max_length):
    sliced = tf.slice(sequence, [0], [min(max_length - 2, sequence.shape[0])])
    enclosed = tf.concat([[pip_tok.BOS_TOK], sliced, [pip_tok.EOS_TOK]], 0)
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        [enclosed],
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
    full_path = os.path.join(
        base_path, str(dnn_type), str(tokenizer_type), f"run_id={str(run_id)}"
    )
    # Create the path if it doesn't exist
    pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
    logging.info(f"Pipeline path (path={full_path})")

    optimizer = transformer_m.optimizer(D_MODEL)
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, full_path, max_to_keep=5)
    if restore:
        if ckpt_manager.latest_checkpoint:
            # if a checkpoint exists, restore the latest checkpoint.
            ckpt.restore(ckpt_manager.latest_checkpoint)
            logging.info(f"Latest checkpoint restored (path={full_path})")
        else:
            logging.info(f"No checkpoing found (path={full_path})")

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

    def __call__(self, sentence, real):
        # Input sentence is the EHR, hence preparing with BOS and EOS
        sequence = self.tok_x.tokenize(sentence)
        encoder_input = normalize(sequence, X_MAX_LEN)[tf.newaxis]

        real = self.tok_y.tokenize(real)
        real_nom = normalize(real, Y_MAX_LEN)

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
    for (i, (x, y)) in enumerate(train_xy):
        print("-" * 80)
        print("START")
        print(f"Iteration (i={i})")
        # print(f"Input X={x}")
        print(f"Input Y={y}")
        translated_text, translated_tokens, attention_weights = translator(x, y)
        print(f"text={translated_text}, tokens={translated_tokens}")
        print("END")
    return


def train(run_id, dnn_type, tokenizer_type, train_xy, test_xy, **kwargs):
    if dnn_type == DnnType.XFMR_XFMR:
        train_xfmr_xfmr(run_id, tokenizer_type, train_xy, test_xy, **kwargs)


def train_xfmr_xfmr(run_id, tokenizer_type, train_xy, test_xy, restore=True, save=True):
    logging.info(
        f"Training DNN (run_id={run_id}, dnn={DnnType.XFMR_XFMR},"
        f"tok={tokenizer_type})"
    )
    train_x, train_y = pip.unpack_2d(train_xy)

    logging.info("Restoring tokenizer...")
    tok_x, tok_y = pip_tok.restore(run_id, tokenizer_type)

    # DEBUG - Fixed elements
    debug_x = []
    debug_y = []
    for i, (x, y) in enumerate(zip(train_x, train_y)):
       debug_x.append(x)
       debug_y.append(y)
       if i >= (1024 - 1):
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

    logging.info("Restoring dnn...")
    transformer, optimizer, ckpt_manager = restore_or_init(
        run_id, DnnType.XFMR_XFMR, tokenizer_type, restore=restore
    )

    # DEBUG
    translator = Translator(
        run_id, DnnType.XFMR_XFMR, tokenizer_type, transformer=transformer
    )
    for vals in zip(train_x, train_y):
        first_x, first_y = vals
        break

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
        if train_accuracy.result() >= ACCURACY_TH:
            logging.info(
                f"Accuracy reached: Epoch {epoch} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}"
            )
            break

        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> ehr, tar -> icd classification
        for (batch, (inp, tar)) in enumerate(train_batches):
            train_step(inp, tar)

            if batch > 0 and batch % 50 == 0:
                # text, tokens, _ = translator(first_x, first_y)
                # print("-" * 80)
                # print("Real")
                # print(first_y)
                # print(tok_y.tokenize(first_y))
                # print("Predicted")
                # print(text)
                # print(tokens)

                logging.info(
                    f"Epoch {epoch} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}"
                )
                logging.info(f"Elapsed time: {time.time() - start:.2f} secs")

        if save:
            ckpt_save_path = ckpt_manager.save()
            logging.info(f"Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}")

        logging.info(
            f"Epoch {epoch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}"
        )

        logging.info(f"Time taken for 1 epoch: {time.time() - start:.2f} secs\n")

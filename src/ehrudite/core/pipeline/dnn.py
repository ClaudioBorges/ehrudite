"""ehrudite tokenizer pipeline"""


from ehrudite.core.pipeline import BASE_PATH
from ehrudite.core.pipeline import make_progressable
from ehrudite.core.pipeline import unpack_2d
import ehrudite.core.dnn.transformer as transformer_m
import ehrudite.core.pipeline as pip
import ehrudite.core.pipeline.tokenizer as pip_tok
import ehrudite.core.text as er_text
import logging
import os
import tensorflow as tf
import time


MODEL_CHECKPOINT_BASE_PATH = os.path.join(BASE_PATH, "model/checkpoint/train/")
MODEL_XFMR_XFMR_BASE_PATH = os.path.join(MODEL_CHECKPOINT_BASE_PATH, "xfmr_xfmr/")
MODEL_XFMR_LSTM_BASE_PATH = os.path.join(MODEL_CHECKPOINT_BASE_PATH, "xfmr_lstm/")
MODEL_LSTM_LSTM_BASE_PATH = os.path.join(MODEL_CHECKPOINT_BASE_PATH, "lstm_lstm/")
MODEL_LSTM_XFMR_BASE_PATH = os.path.join(MODEL_CHECKPOINT_BASE_PATH, "lstm_xfmr/")


X_MAX_LEN = 1024
Y_MAX_LEN = 128
BUFFER_SIZE = 128  # 20000
BATCH_SIZE = 64
EPOCHS = 20


def train_xfmr_xfmr(run_id, tokenizer_type, train_xy, test_xy):
    train_x, train_y = unpack_2d(train_xy)
    tok_x, tok_y = pip_tok.restore(run_id, tokenizer_type)

    def prepare_xy():
        def normalize(sequence, maxlen):
            sliced = tf.slice(sequence, [0], [min(maxlen - 2, sequence.shape[0])])
            enclosed = tf.concat([[pip_tok.BOS_TOK], sliced, [pip_tok.EOS_TOK]], 0)
            padded = tf.keras.preprocessing.sequence.pad_sequences(
                [enclosed],
                maxlen=maxlen,
                padding="post",
                truncating="post",
                value=pip_tok.PAD_TOK,
            )[0]
            return tf.constant(padded)

        train_x_tok = (tok_x.tokenize(x) for x in train_x)
        train_y_tok = (tok_y.tokenize(y) for y in train_y)

        return (
            (normalize(x, X_MAX_LEN), normalize(y, Y_MAX_LEN))
            for x, y in zip(train_x_tok, train_y_tok)
        )

    train_xy_tok_gen = er_text.LenghtableRepeatableGenerator(
        prepare_xy, _length=len(train_xy)
    )

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
        input_vocab_size=tok_x.vocab_size(),
        target_vocab_size=tok_y.vocab_size(),
        pe_input=X_MAX_LEN,
        pe_target=Y_MAX_LEN,
        rate=dropout_rate,
    )

    optimizer = transformer_m.optimizer(d_model)

    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(
        ckpt, MODEL_XFMR_XFMR_BASE_PATH, max_to_keep=5
    )

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!!")

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
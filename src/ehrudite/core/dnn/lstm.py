"""Long Short-Term Memory (LSTM)"""

import numpy as np
import tensorflow as tf


class Transformer(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        target_vocab_size,
        pe_input,
        pe_target,
        rate=0.1,
    ):
        super().__init__()
        self.encoder = Encoder(
            num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate
        )

        self.decoder = Decoder(
            num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate
        )

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs

        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(
            inp, tar
        )

        enc_output = self.encoder(
            inp, training, enc_padding_mask
        )  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask
        )

        final_output = self.final_layer(
            dec_output
        )  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output, attention_weights

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = _create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = _create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = _create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = _create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, look_ahead_mask, dec_padding_mask


class Encoder(tf.keras.layers.Layer):
    """Bi-LSTM encoder

    Contains:
        1. Input Embedding
        3. N encoder layers
    """

    def __init__(
        self,
        num_layers,
        n_hidden,
        input_vocab_size,
    ):
        super(Encoder, self).__init__()

        self.n_hidden = n_hidden
        assert n_hidden % 2 == 0, "Hidden state must be divisable by 2"

        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, n_hidden)
        self.enc_layers = [
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    int(n_hidden / 2), return_sequences=True, return_state=True
                )
            )
            for _ in range(num_layers)
        ]

    def call(self, x, training, return_extras=False):
        x = self.embedding(x)  # (batch_size, input_seq_len, n_hidden)

        state_hs = []
        state_cs = []
        for i in range(self.num_layers):
            x, forward_h, forward_c, backward_h, backward_c = self.enc_layers[i](
                x, training=training
            )
            state_hs.append(tf.keras.layers.Concatenate()([forward_h, backward_h]))
            state_cs.append(tf.keras.layers.Concatenate()([forward_c, backward_c]))

        if return_extras is True:
            return x, state_hs, state_cs
        return x  # (batch_size, input_seq_len, n_hidden * 2)


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        target_vocab_size,
        maximum_position_encoding,
        rate=0.1,
    ):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = _positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x, enc_output, training, look_ahead_mask, padding_mask
            )

        attention_weights[f"decoder_layer{i+1}_block1"] = block1
        attention_weights[f"decoder_layer{i+1}_block2"] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


def optimizer(d_model):
    """Adam optimizer as of Section 5.3"""

    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=4000):
            super(CustomSchedule, self).__init__()

            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)

            self.warmup_steps = warmup_steps

        def __call__(self, step):
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)

            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    learning_rate = CustomSchedule(d_model)
    return tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

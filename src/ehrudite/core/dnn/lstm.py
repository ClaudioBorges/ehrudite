"""Long Short-Term Memory (LSTM)"""
# Reference: https://www.tensorflow.org/text/tutorials/nmt_with_attention

import ehrudite.core.helper as er_helper
import ehrudite.core.pipeline.tokenizer as pip_tok
import numpy as np
import tensorflow as tf


class Seq2SeqBiLstmAttn(tf.keras.Model):
    def __init__(
        self,
        embedding_dim,
        units,
        input_vocab_size,
        target_vocab_size,
    ):
        super().__init__()
        self.encoder = Encoder(input_vocab_size, embedding_dim, units)
        self.decoder = Decoder(target_vocab_size, embedding_dim, units)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        self.shape_checker = er_helper.ShapeChecker()

    def call(self, inputs, training, enc_output=None, dec_initial_states=None):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs
        self.shape_checker(inp, ("batch", "s"))
        self.shape_checker(tar, ("batch", "t"))

        input_mask = self.padding_mask(inp)
        target_mask = self.padding_mask(tar)

        if enc_output is None:
            enc_output, enc_states = self.encoder(
                inp, training, padding_mask=input_mask
            )
            self.shape_checker(enc_output, ("batch", "s", "enc_units"))
            self.shape_checker(enc_states[0], ("batch", "enc_units"))
            self.shape_checker(enc_states[1], ("batch", "enc_units"))
            # Set Decoder initial state as Encoder (as original paper)
            dec_initial_states = enc_states

        dec_attn_vector, dec_attn_weights, dec_states = self.decoder(
            tar,
            enc_output,
            enc_padding_mask=input_mask,
            dec_padding_mask=target_mask,
            initial_state=dec_initial_states,
        )
        self.shape_checker(dec_attn_vector, ("batch", "t", "dec_units"))
        self.shape_checker(dec_attn_weights, ("batch", "t", "s"))
        self.shape_checker(dec_states[0], ("batch", "enc_units"))
        self.shape_checker(dec_states[1], ("batch", "enc_units"))

        logits = self.final_layer(dec_attn_vector)
        self.shape_checker(logits, ("batch", "t", "output_vocab_size"))

        return logits, dec_attn_weights, enc_output, dec_states

    def padding_mask(self, sequence):
        self.shape_checker(sequence, ("batch", "n"))
        masked = sequence != pip_tok.PAD_TOK
        self.shape_checker(masked, ("batch", "n"))
        return masked


class Encoder(tf.keras.layers.Layer):
    def __init__(self, input_vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()

        self.shape_checker = er_helper.ShapeChecker()
        self.enc_units = enc_units
        assert enc_units % 2 == 0, "Hidden state must be divisable by 2"

        # The embedding layer converts tokens to vectors
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, embedding_dim)

        # The RNN layer processes those vectors sequentially.
        self.bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                int(enc_units / 2), return_sequences=True, return_state=True
            )
        )

    def call(self, x, training, padding_mask=None, initial_state=None):
        shape_checker = er_helper.ShapeChecker()
        shape_checker(x, ("batch", "s"))

        # 2. The embedding layer looks up the embedding for each token.
        vectors = self.embedding(x)
        shape_checker(vectors, ("batch", "s", "embed_dim"))

        # 3. The RNN processes the embedding sequence.
        output, forward_h, forward_c, backward_h, backward_c = self.bi_lstm(
            vectors, training=training, mask=padding_mask, initial_state=initial_state
        )
        state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
        state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])

        shape_checker(output, ("batch", "s", "enc_units"))
        shape_checker(state_h, ("batch", "enc_units"))
        shape_checker(state_c, ("batch", "enc_units"))

        # 4. Returns the new sequence and its state.
        return output, (state_h, state_c)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()

        self.shape_checker = er_helper.ShapeChecker()
        self.dec_units = dec_units
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim

        # For Step 1. The embedding layer convets token IDs to vectors
        self.embedding = tf.keras.layers.Embedding(
            self.output_vocab_size, embedding_dim
        )

        # For Step 2. The RNN keeps track of what's been generated so far.
        self.lstm = tf.keras.layers.LSTM(
            self.dec_units,
            return_sequences=True,
            return_state=True,
        )  # recurrent_initializer='glorot_uniform')

        # For step 3. The RNN output will be the query for the attention layer.
        self.attention = BahdanauAttention(self.dec_units)

        # For step 4. Eqn. (3): converting `ct` to `at`
        self.Wc = tf.keras.layers.Dense(
            dec_units, activation=tf.math.tanh, use_bias=False
        )

    def call(
        self,
        x,
        enc_output,
        enc_padding_mask=None,
        dec_padding_mask=None,
        initial_state=None,
    ):
        self.shape_checker = er_helper.ShapeChecker()
        self.shape_checker(x, ("batch", "t"))
        self.shape_checker(enc_output, ("batch", "s", "enc_units"))
        self.shape_checker(enc_padding_mask, ("batch", "s"))

        if initial_state is not None:
            self.shape_checker(initial_state[0], ("batch", "dec_units"))
            self.shape_checker(initial_state[1], ("batch", "dec_units"))

        # Step 1. Lookup the embeddings
        vectors = self.embedding(x)
        self.shape_checker(vectors, ("batch", "t", "embedding_dim"))

        # Step 2. Process one step with the RNN
        rnn_output, state_h, state_c = self.lstm(
            vectors, mask=dec_padding_mask, initial_state=initial_state
        )

        self.shape_checker(rnn_output, ("batch", "t", "dec_units"))
        self.shape_checker(state_h, ("batch", "dec_units"))
        self.shape_checker(state_c, ("batch", "dec_units"))

        # Step 3. Use the RNN output as the query for the attention over the
        # encoder output.
        context_vector, attn_weights = self.attention(
            query=rnn_output, value=enc_output, mask=enc_padding_mask
        )
        self.shape_checker(context_vector, ("batch", "t", "dec_units"))
        self.shape_checker(attn_weights, ("batch", "t", "s"))

        # Step 4. Eqn. (3): Join the context_vector and rnn_output
        #     [ct; ht] shape: (batch t, value_units + query_units)
        context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)

        # Step 4. Eqn. (3): `at = tanh(Wc@[ct; ht])`
        attn_vector = self.Wc(context_and_rnn_output)
        self.shape_checker(attn_vector, ("batch", "t", "dec_units"))

        return attn_vector, attn_weights, (state_h, state_c)


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        # For Eqn. (4), the  Bahdanau attention
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)

        self.attention = tf.keras.layers.AdditiveAttention()

    def call(self, query, value, mask):
        self.shape_checker = er_helper.ShapeChecker()
        self.shape_checker(query, ("batch", "t", "query_units"))
        self.shape_checker(value, ("batch", "s", "value_units"))
        self.shape_checker(mask, ("batch", "s"))

        # From Eqn. (4), `W1@ht`.
        w1_query = self.W1(query)
        self.shape_checker(w1_query, ("batch", "t", "attn_units"))

        # From Eqn. (4), `W2@hs`.
        w2_key = self.W2(value)
        self.shape_checker(w2_key, ("batch", "s", "attn_units"))

        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        value_mask = mask

        context_vector, attn_weights = self.attention(
            inputs=[w1_query, value, w2_key],
            mask=[query_mask, value_mask],
            return_attention_scores=True,
        )
        self.shape_checker(context_vector, ("batch", "t", "value_units"))
        self.shape_checker(attn_weights, ("batch", "t", "s"))

        return context_vector, attn_weights

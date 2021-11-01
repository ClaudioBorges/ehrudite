"""The test file for LSTM DNN"""

import ehrudite.core.dnn.lstm as lstm
import os
import random
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def test_transformer_encoder_decoder():
    batch_size = random.randint(8, 64)
    rnn_units = 2 ** random.randint(7, 9)
    input_seq_len = random.randint(40, 50)
    input_vocab_size = random.randint(1000, 10000)
    embedding_dim = 2 ** random.randint(7, 9)
    target_seq_len = random.randint(10, 30)
    target_vocab_size = random.randint(1000, 10000)

    encoder = lstm.Encoder(
        input_vocab_size,
        embedding_dim,
        rnn_units,
    )
    temp_input = tf.random.uniform(
        (batch_size, input_seq_len), dtype=tf.int64, minval=0, maxval=200
    )
    enc_output, enc_states = encoder(temp_input, training=False)

    assert enc_output.shape == (batch_size, input_seq_len, rnn_units)
    assert enc_states[0].shape == (batch_size, rnn_units)
    assert enc_states[1].shape == (batch_size, rnn_units)

    decoder = lstm.Decoder(target_vocab_size, embedding_dim, rnn_units)
    temp_target = tf.random.uniform(
        (batch_size, target_seq_len), dtype=tf.int64, minval=0, maxval=200
    )
    mask = tf.ones([batch_size, input_seq_len])
    dec_attn_vector, dec_attn_weights, dec_states = decoder(
        temp_target,
        enc_output,
        temp_input != 0,
        initial_state=enc_states,
    )

    assert dec_attn_vector.shape == (batch_size, target_seq_len, rnn_units)
    assert dec_attn_weights.shape == (
        batch_size,
        target_seq_len,
        input_seq_len,
    )
    assert dec_states[0].shape == (batch_size, rnn_units)
    assert dec_states[1].shape == (batch_size, rnn_units)


def test_seq2seq_bilstm_attn():
    batch_size = random.randint(8, 64)
    rnn_units = 2 ** random.randint(7, 9)
    input_seq_len = random.randint(40, 50)
    input_vocab_size = random.randint(1000, 10000)
    embedding_dim = 2 ** random.randint(7, 9)
    target_seq_len = random.randint(10, 30)
    target_vocab_size = random.randint(1000, 10000)

    model = lstm.Seq2SeqBiLstmAttn(
        embedding_dim, rnn_units, input_vocab_size, target_vocab_size
    )

    temp_input = tf.random.uniform(
        (batch_size, input_seq_len), dtype=tf.int64, minval=0, maxval=200
    )
    temp_target = tf.random.uniform(
        (batch_size, target_seq_len), dtype=tf.int64, minval=0, maxval=200
    )

    output, _ = model([temp_input, temp_target], training=False)
    assert fn_out.shape == (batch_size, target_seq_len, target_vocab_size)

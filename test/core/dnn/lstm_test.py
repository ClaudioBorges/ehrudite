"""The test file for LSTM DNN"""

import ehrudite.core.dnn.lstm as lstm
import os
import random
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def test_transformer_encoder_decoder():
    batch_size = random.randint(8, 64)
    n_hidden = 2 ** random.randint(7, 9)
    input_seq_len = random.randint(40, 50)
    input_vocab_size = random.randint(1000, 10000)
    num_layers = 4  # random.randint(2, 4)
    # target_seq_len = random.randint(10, 30)
    # target_vocab_size = random.randint(1000, 10000)

    encoder = lstm.Encoder(
        num_layers=num_layers,
        n_hidden=n_hidden,
        input_vocab_size=input_vocab_size,
    )
    temp_input = tf.random.uniform(
        (batch_size, input_seq_len), dtype=tf.int64, minval=0, maxval=200
    )
    encoder_output, _, _ = encoder(temp_input, training=False, return_extras=True)

    assert encoder_output.shape == (batch_size, input_seq_len, n_hidden)

    # decoder = transformer.Decoder(
    #    num_layers=num_layers,
    #    d_model=d_model,
    #    num_heads=num_heads,
    #    dff=dff,
    #    target_vocab_size=target_vocab_size,
    #    maximum_position_encoding=maximum_position_encoding,
    # )
    # temp_input = tf.random.uniform(
    #    (batch_size, target_seq_len), dtype=tf.int64, minval=0, maxval=200
    # )
    # output, attn = decoder(
    #    temp_input,
    #    enc_output=encoder_output,
    #    training=False,
    #    look_ahead_mask=None,
    #    padding_mask=None,
    # )

    # assert output.shape == (batch_size, target_seq_len, d_model)
    # assert len(attn.keys()) == 2


# def test_lstm_model():
# batch_size = random.randint(8, 64)
# d_model = 2 ** random.randint(7, 9)
# dff = random.randint(512, 2048)
# input_seq_len = random.randint(40, 50)
# input_vocab_size = random.randint(1000, 10000)
# num_heads = 2 ** random.randint(1, 4)
# num_layers = random.randint(2, 4)
# target_seq_len = random.randint(10, 30)
# target_vocab_size = random.randint(1000, 10000)

# sample_transformer = transformer.Transformer(
#     num_layers=num_layers,
#     d_model=d_model,
#     num_heads=num_heads,
#     dff=dff,
#     input_vocab_size=input_vocab_size,
#     target_vocab_size=target_vocab_size,
#     pe_input=random.randint(5000, 10000),
#     pe_target=random.randint(2000, 4000),
# )

# temp_input = tf.random.uniform(
#     (batch_size, input_seq_len), dtype=tf.int64, minval=0, maxval=200
# )
# temp_target = tf.random.uniform(
#     (batch_size, target_seq_len), dtype=tf.int64, minval=0, maxval=200
# )

# fn_out, _ = sample_transformer([temp_input, temp_target], training=False)
# assert fn_out.shape == (batch_size, target_seq_len, target_vocab_size)


# def test_optimizer():
#    d_model = 2 ** random.randint(7, 9)
#    optimizer = transformer.optimizer(d_model)
#    assert optimizer is not None

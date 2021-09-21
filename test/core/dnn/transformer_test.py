"""The test file for transformer DNN"""

import ehrudite.core.dnn.transformer as transformer
import random
import tensorflow as tf


def test_transformer_encoder_decoder_layer():
    batch_size = random.randint(8, 64)
    d_model = random.randint(128, 512)
    dff = random.randint(512, 2048)
    input_seq_len = random.randint(40, 50)
    target_seq_len = random.randint(10, 30)
    num_heads = random.randint(2, 8)

    encoder_layer = transformer.EncoderLayer(d_model, num_heads, dff)
    encoder_layer_output = encoder_layer(
        tf.random.uniform((batch_size, input_seq_len, d_model)), False, None
    )
    assert (batch_size, input_seq_len, d_model) == encoder_layer_output.shape

    decoder_layer = transformer.DecoderLayer(d_model, num_heads, dff)
    decoder_layer_output, _, _ = decoder_layer(
        tf.random.uniform((batch_size, target_seq_len, d_model)),
        encoder_layer_output,
        False,
        None,
        None,
    )
    assert (batch_size, target_seq_len, d_model) == decoder_layer_output.shape


def test_transformer_encoder_decoder():
    batch_size = random.randint(8, 64)
    d_model = random.randint(128, 512)
    dff = random.randint(512, 2048)
    input_seq_len = random.randint(40, 50)
    input_vocab_size = random.randint(1000, 10000)
    maximum_position_encoding = random.randint(1024, 4096)
    num_heads = random.randint(2, 8)
    num_layers = random.randint(2, 4)
    target_seq_len = random.randint(10, 30)
    target_vocab_size = random.randint(1000, 10000)

    encoder = transformer.Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=input_vocab_size,
        maximum_position_encoding=maximum_position_encoding,
    )
    temp_input = tf.random.uniform(
        (batch_size, input_seq_len), dtype=tf.int64, minval=0, maxval=200
    )
    encoder_output = encoder(temp_input, training=False, mask=None)

    assert encoder_output.shape == (batch_size, input_seq_len, d_model)

    decoder = transformer.Decoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        target_vocab_size=target_vocab_size,
        maximum_position_encoding=maximum_position_encoding,
    )
    temp_input = tf.random.uniform(
        (batch_size, target_seq_len), dtype=tf.int64, minval=0, maxval=200
    )
    output, attn = decoder(
        temp_input,
        enc_output=encoder_output,
        training=False,
        look_ahead_mask=None,
        padding_mask=None,
    )

    assert output.shape == (batch_size, target_seq_len, d_model)
    assert len(attn.keys()) == 2


def test_transformer_positional_encoding():
    maximum_position_encoding = random.randint(1024, 4096)
    d_model = random.randint(128, 512)
    pos_encoding = transformer._positional_encoding(maximum_position_encoding, d_model)
    assert pos_encoding.shape == (1, maximum_position_encoding, d_model)


def test_transformer_model():
    batch_size = random.randint(8, 64)
    d_model = random.randint(128, 512)
    dff = random.randint(512, 2048)
    input_seq_len = random.randint(40, 50)
    input_vocab_size = random.randint(1000, 10000)
    num_heads = random.randint(2, 8)
    num_layers = random.randint(2, 4)
    target_seq_len = random.randint(10, 30)
    target_vocab_size = random.randint(1000, 10000)

    sample_transformer = transformer.Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        pe_input=random.randint(5000, 10000),
        pe_target=random.randint(2000, 4000),
    )

    temp_input = tf.random.uniform(
        (batch_size, input_seq_len), dtype=tf.int64, minval=0, maxval=200
    )
    temp_target = tf.random.uniform(
        (batch_size, target_seq_len), dtype=tf.int64, minval=0, maxval=200
    )

    fn_out, _ = sample_transformer([temp_input, temp_target], training=False)
    assert fn_out.shape == (batch_size, target_seq_len, target_vocab_size)

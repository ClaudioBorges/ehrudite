"""Transformer from 'Attention is all you need' (Vaswani et. al., 2017)"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TransformerEncoderLayer(layers.Layer):
    """Transformer encoder from 'Attention is all you need' (Vaswani et. al., 2017)

    One of the main difference between the transformer encoder from decoder is
    the self-attention. The reasons for it is detailed in the Section 4 and can
    be summarized as a way to reduce the path length between long-range depencies
    in the network.
    """

    def __init__(self, d_model=512, num_heads=8, d_ff=2048, p_drop=0.1):
        """Initializer a Transformer Encoder Layer

        Attributes
        ----------
        d_model : int
            Model dimension used on all sub-layers and embedding.
        num_heads : int
            Number of heads. Vaswani et. al., 2017 describes as $h$
        d_ff : int
            FeedForward dimension.
        p_drop : float
            Dropout rate parameter applied after self-attention and
            FeedForward.
        """

        super(TransformerEncoderLayer, self).__init__()
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        # Position-wise Feed-Forward Network (Section 3.3)
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        self.ffn = keras.Sequential(
            [
                layers.Dense(d_ff, activation="relu"),
                layers.Dense(d_model),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(p_drop)
        self.dropout2 = layers.Dropout(p_drop)

    def call(self, inputs, training):
        # 1. Self-Attention
        attn_output = self.attn(inputs, inputs)
        # 2. Residual Dropout
        attn_output = self.dropout1(attn_output, training=training)
        # 3. Add & Norm
        out1 = self.layernorm1(inputs + attn_output)
        # 4. FeedForward
        ffn_output = self.ffn(out1)
        # 5. Residual Dropout
        ffn_output = self.dropout2(ffn_output, training=training)
        # 6. Add & Norm
        return self.layernorm2(out1 + ffn_output)


class TransformerInputEmbeddingLayer(layers.Layer):
    def __init__(self, maxlen, vocab_size, d_model):
        super(TransformerInputEmbeddingLayer, self).__init__()
        self._d_model = d_model
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=d_model)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=d_model)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        # Multiply the embedding layer to sqrt(d_model) as per Section 3.4
        x = self.token_emb(x) * tf.math.sqrt(float(self._d_model))
        # x = self.token_emb(x)
        return x + positions


vocab_size = 2000
maxlen = 200
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

d_model = 512  # Embedding size for each token
num_heads = 6  # Number of attention heads
d_ff = 1024  # Hidden layer size in feed forward network inside transformer
num_transformer_blocks = 6  # Number of transformer blocks

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TransformerInputEmbeddingLayer(maxlen, vocab_size, d_model)
x = embedding_layer(inputs)
for _ in range(num_transformer_blocks):
    transformer_block = TransformerEncoderLayer(d_model, num_heads, d_ff)
    x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(2, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)


model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    x_train, y_train, batch_size=64, epochs=4, validation_data=(x_val, y_val)
)

"""skipgram word2vec from tensorflow embedding"""

from tensorflow.keras import layers
import io
import numpy as np
import os
import tensorflow as tf
import tqdm


class SkipGramModel(tf.keras.Model):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        n_negative_samples=4,
        window_size=4,
        buffer_size=1024,
        batch_size=128,
        epochs=5,
    ):
        super(SkipGramModel, self).__init__()

        self._vocab_size = vocab_size
        self._embedding_dim = embedding_dim
        self._n_negative_samples = n_negative_samples
        self._window_size = window_size
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._epochs = epochs

        self.target_embedding = layers.Embedding(
            vocab_size, embedding_dim, input_length=1, name="w2v_embedding"
        )
        self.context_embedding = layers.Embedding(
            vocab_size, embedding_dim, input_length=self._n_negative_samples + 1
        )

    def call(self, pair):
        target, context = pair
        # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
        # context: (batch, context)
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        # target: (batch,)
        word_emb = self.target_embedding(target)
        # word_emb: (batch, embed)
        context_emb = self.context_embedding(context)
        print(f"My Debug: {word_emb}, {context_emb}")
        # context_emb: (batch, context, embed)
        dots = tf.einsum("be,bce->bc", word_emb, context_emb)
        # dots: (batch, context)
        return dots

    def skipgram_from_sequence_ds(self, sequence_ds):
        sequences = (sequence for sequence in sequence_ds.as_numpy_iterator())
        targets, contexts, labels = self._generate_training_data(sequences)

        targets = np.array(targets)
        contexts = np.array(contexts)[:, :, 0]
        labels = np.array(labels)

        print("\n")
        print(f"targets.shape: {targets.shape}")
        print(f"contexts.shape: {contexts.shape}")
        print(f"labels.shape: {labels.shape}")

        dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
        dataset = dataset.shuffle(self._buffer_size).batch(
            self._batch_size, drop_remainder=True
        )
        dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def auto_compile(self):
        self.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def run(self, sequence_ds, seed=None):
        skipgram_ds = self.skipgram_from_sequence_ds(sequence_ds)
        self.auto_compile()
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
        self.fit(skipgram_ds, epochs=self._epochs, callbacks=[tensorboard_callback])

    # def save(self, folder='.'):
    #    out_v = io.open(os.path.join(folder, 'vectors.tsv'), 'w', encoding='utf-8')
    #    weights = self.get_layer('w2v_embedding').get_weights()[0]
    #    for index, weight in enumerate(weights):
    #        out_v.write('\t'.join([str(x) for x in weight]) + "\n")
    #    out_v.close()

    # Generates skip-gram pairs with negative sampling for a list of sequences
    # (int-encoded sentences) based on window size, number of negative samples
    # and vocabulary size.
    def _generate_training_data(self, sequences, seed=None):
        # Elements of each training example are appended to these lists.
        targets, contexts, labels = [], [], []

        # Build the sampling table for vocab_size tokens.
        sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(
            self._vocab_size
        )

        # Iterate over all sequences (sentences) in dataset.
        for sequence in tqdm.tqdm(sequences):
            # Generate positive skip-gram pairs for a sequence (sentence).
            positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
                sequence,
                vocabulary_size=self._vocab_size,
                sampling_table=sampling_table,
                window_size=self._window_size,
                negative_samples=0,
            )

            # Iterate over each positive skip-gram pair to produce training examples
            # with positive context word and negative samples.
            for target_word, context_word in positive_skip_grams:
                context_class = tf.expand_dims(
                    tf.constant([context_word], dtype="int64"), 1
                )
                (
                    negative_sampling_candidates,
                    _,
                    _,
                ) = tf.random.log_uniform_candidate_sampler(
                    true_classes=context_class,
                    num_true=1,
                    num_sampled=self._n_negative_samples,
                    unique=True,
                    range_max=self._vocab_size,
                    seed=seed,
                    name="negative_sampling",
                )

                # Build context and label vectors (for one target word)
                negative_sampling_candidates = tf.expand_dims(
                    negative_sampling_candidates, 1
                )

                context = tf.concat([context_class, negative_sampling_candidates], 0)
                label = tf.constant([1] + [0] * self._n_negative_samples, dtype="int64")

                # Append each element from the training example to global lists.
                targets.append(target_word)
                contexts.append(context)
                labels.append(label)

        return targets, contexts, labels

"""init py for embedding."""

import collections


class NotTrainedError(Exception):
    pass


class NotFitToCorpusError(Exception):
    pass


EmbeddingEntity = collections.namedtuple(
    "EmbeddingEntity", ["embeddings", "words", "id_for_word"]
)

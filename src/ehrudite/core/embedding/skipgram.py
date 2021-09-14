"""Implement Skipgram from Word2Vec"""

from ehrudite.core.embedding import NotFitToCorpusError
from ehrudite.core.embedding import NotTrainedError
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


class MonitorCallback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print("Loss after epoch {}: {}".format(self.epoch, loss))
        else:
            print(
                "Loss after epoch {}: {}".format(
                    self.epoch, loss - self.loss_previous_step
                )
            )
        self.epoch += 1
        self.loss_previous_step = loss


class SkipgramModel:
    def __init__(self, embedding_size=128, context_size=5):
        self._model = None
        self._corpus = None
        self._embedding_size = embedding_size
        self._context_size = context_size

    def fit_to_corpus(self, corpus):
        self._corpus = corpus

    def train(
        self,
        negative=4,
        workers=4,
        ns_exponent=0.75,
        min_count=1,
        num_epochs=5,
        **kwargs
    ):
        if self._corpus is None:
            raise NotFitToCorpusError("Need to fit model to corpus before training.")

        self._model = Word2Vec(
            callbacks=[MonitorCallback()],
            compute_loss=True,
            epochs=num_epochs,
            min_count=min_count,
            negative=negative,
            ns_exponent=ns_exponent,
            sentences=self._corpus,
            sg=1,
            vector_size=self._embedding_size,
            window=self._context_size,
            workers=workers,
            **kwargs,
        )

    def embedding_for_word(self, word):
        if self._model is None:
            raise NotTrainedError(
                "Need to train model before accessing embedding_for_word"
            )
        return self.embedding_for_id(self.id_for_word(word))

    def embedding_for_id(self, model_id):
        if self._model is None:
            raise NotTrainedError(
                "Need to train model before accessing embedding_for_id"
            )
        return self._model.wv.vectors[model_id]

    @property
    def vocab_size(self):
        if self._model is None:
            raise NotTrainedError("Need to train model before accessing vocab_size")
        return len(self._model.wv.index_to_key)

    @property
    def words(self):
        if self._model is None:
            raise NotTrainedError("Need to train model before accessing words")
        return self._model.wv.index_to_key

    @property
    def embeddings(self):
        if self._model is None:
            raise NotTrainedError("Need to train model before accessing embeddings")
        return self._model.wv.vectors

    def id_for_word(self, word):
        if self._model is None:
            raise NotTrainedError("Need to train model before accessing id_for_word")
        return self._model.wv.key_to_index[word]

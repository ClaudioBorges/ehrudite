"""Implement Skipgram from Word2Vec"""

from ehrudite.core.embedding import EmbeddingEntity
from ehrudite.core.embedding import NotFitToCorpusError
from ehrudite.core.embedding import NotTrainedError
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import pickle


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
        self._embedding_entity = None

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

        self._embedding_entity = EmbeddingEntity(
            self._model.wv.vectors,
            self._model.wv.index_to_key,
            self._model.wv.key_to_index,
        )

    def embedding_for_word(self, word):
        return self.embedding_for_id(self.id_for_word(word))

    def embedding_for_id(self, model_id):
        return self.embeddings[model_id]

    @property
    def vocab_size(self):
        self.__validate_train()
        return len(self.words)

    @property
    def words(self):
        self.__validate_train()
        return self._embedding_entity.words

    @property
    def embeddings(self):
        self.__validate_train()
        return self._embedding_entity.embeddings

    def id_for_word(self, word):
        self.__validate_train()
        return self._embedding_entity.id_for_word[word]

    def save(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self._embedding_entity, f)

    @staticmethod
    def load(file_name):
        with open(file_name, "rb") as f:
            model = SkipgramModel()
            model._embedding_entity = pickle.load(f)
            return model

    def __validate_train(self):
        if self._embedding_entity is None:
            raise NotTrainedError("Need to train model before accessing the model")

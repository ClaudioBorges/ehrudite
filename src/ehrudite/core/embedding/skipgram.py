"""Implement Skipgram from Word2Vec"""

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


class Skipgram:
    def __init__(self):
        self._model = None

    def train(
        self,
        repeatable_generator,
        embedding_dim=128,
        negative=4,
        window=4,
        workers=4,
        ns_exponent=0.75,
        sg=1,
        min_count=1,
        **kwargs
    ):
        self._model = Word2Vec(
            callbacks=[MonitorCallback()],
            compute_loss=True,
            min_count=min_count,
            negative=negative,
            ns_exponent=ns_exponent,
            sentences=repeatable_generator,
            sg=sg,
            vector_size=embedding_dim,
            window=window,
            workers=workers,
            **kwargs,
        )

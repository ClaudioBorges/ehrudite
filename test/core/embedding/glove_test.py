"""The test file for Glove embedding"""

from ehrudite.core.embedding import NotFitToCorpusError
from ehrudite.core.embedding import NotTrainedError
import ehrudite.core.embedding.glove as glove
import pytest
import random
import test.core.embedding as t_embedding


def _make_random_trained_model():
    corpus, corpus_vocab = t_embedding.random_corpus()

    context_size = random.randint(2, 5)
    embedding_size = random.randint(10, 15)
    epochs = random.randint(2, 4)

    model = glove.GloveModel(embedding_size=embedding_size, context_size=context_size)
    model.fit_to_corpus(corpus)
    model.train(num_epochs=epochs)

    return model, corpus_vocab, embedding_size


def test_glove_embedding():
    model, corpus_vocab, embedding_size = _make_random_trained_model()
    t_embedding.assert_model(model, corpus_vocab, embedding_size)


def test_skipgram_embedding_save_load(tmp_path):
    model, corpus_vocab, embedding_size = _make_random_trained_model()

    file = tmp_path / "glove.embedding"
    model.save(file)
    new_model = glove.GloveModel.load(file)

    t_embedding.assert_model_equals(model, new_model)


def test_glove_not_trained_error_on_embeddings():
    model = glove.GloveModel(embedding_size=1, context_size=1)
    with pytest.raises(NotTrainedError):
        model.embeddings()


def test_glove_not_trained_error_on_words():
    model = glove.GloveModel(embedding_size=1, context_size=1)
    with pytest.raises(NotTrainedError):
        model.words()


def test_glove_not_trained_error_on_id_to_words():
    model = glove.GloveModel(embedding_size=1, context_size=1)
    with pytest.raises(NotTrainedError):
        model.id_for_word(0)


def test_glove_not_fit_error_on_train():
    model = glove.GloveModel(embedding_size=1, context_size=1)
    with pytest.raises(NotFitToCorpusError):
        model.train(1)

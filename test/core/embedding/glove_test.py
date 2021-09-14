"""The test file for Glove embedding"""

import ehrudite.core.embedding.glove as glove
import random


def _random_sentence(vocab, len_min=1, len_max=25):
    return [random.choice(vocab) for _ in range(random.randint(len_min, len_max))]


def _random_corpus(vocab, sentences_min=50, sentences_max=100):
    return [
        _random_sentence(vocab)
        for _ in range(random.randint(sentences_min, sentences_max))
    ]


def _random_vocab(samples, id_min=5, id_max=5000):
    return list(set([random.randint(id_min, id_max) for i in range(samples)]))


def test_glove_embedding():
    vocab = _random_vocab(random.randint(10, 500))
    corpus = _random_corpus(vocab)
    corpus_vocab = sorted(list(set([word for sentence in corpus for word in sentence])))
    corpus_vocab_size = len(corpus_vocab)

    context_size = random.randint(2, 5)
    embedding_size = random.randint(10, 15)
    epochs = random.randint(2, 4)

    model = glove.GloVeModel(embedding_size=embedding_size, context_size=context_size)
    model.fit_to_corpus(corpus)
    model.train(num_epochs=epochs)

    assert model.vocab_size == corpus_vocab_size
    assert len(model.embeddings) == corpus_vocab_size
    assert sorted(model.words) == corpus_vocab

    for word in corpus_vocab:
        assert len(model.embedding_for_word(word)) == embedding_size
    for model_id in range(corpus_vocab_size):
        assert len(model.embedding_for_id(model_id)) == embedding_size

    for word in corpus_vocab:
        assert len(model.embedding_for_id(model.id_for_word(word))) == embedding_size

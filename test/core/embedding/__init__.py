"""init py for test."""

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


def random_corpus():
    vocab = _random_vocab(random.randint(10, 500))
    corpus = _random_corpus(vocab)
    corpus_vocab = sorted(list(set([word for sentence in corpus for word in sentence])))
    return corpus, corpus_vocab


def assert_model(model, corpus_vocab, embedding_size):
    corpus_vocab_size = len(corpus_vocab)
    assert model.vocab_size == corpus_vocab_size
    assert len(model.embeddings) == corpus_vocab_size
    assert sorted(model.words) == corpus_vocab

    for word in corpus_vocab:
        assert len(model.embedding_for_word(word)) == embedding_size
    for model_id in range(corpus_vocab_size):
        assert len(model.embedding_for_id(model_id)) == embedding_size

    for word in corpus_vocab:
        assert len(model.embedding_for_id(model.id_for_word(word))) == embedding_size


def assert_model_equals(model1, model2):
    assert model1.embeddings.tolist() == model2.embeddings.tolist()
    assert model1.vocab_size == model2.vocab_size
    assert model1.words == model2.words

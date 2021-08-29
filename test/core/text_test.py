"""The test file for text"""

import ehrudite.core.text as er_text


def test_generator_callable():
    iterator = (i for i in range(1))
    generator = er_text.Generator(iterator)
    assert callable(generator) == True


def test_generator_content():
    elms = [i for i in range(10)]
    iterator = (i for i in elms)
    generator = er_text.Generator(iterator)
    assert list(generator()) == elms


def test_texts_to_sentence():
    elms = ["\nA\nb\n", "C"]

    got = er_text.texts_to_sentences(elms, to_lower=False)
    expected = ["", "A", "b", "C"]
    assert list(got) == expected

    got_lower = er_text.texts_to_sentences(elms)
    expected_lower = list(map(lambda s: s.lower(), expected))
    assert list(got_lower) == expected_lower


def test_sentences_to_words():
    elms = ["sentence one", "part 2."]
    got = er_text.sentences_to_words(elms)
    expected = ["sentence", "one", "part", "2."]
    assert list(got) == expected

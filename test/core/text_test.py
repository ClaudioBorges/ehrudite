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


def test_repeatable_generator():
    def f_generator(limit):
        return (i for i in range(limit))

    generator = er_text.RepeatableGenerator(f_generator, limit=10)
    assert list(generator()) == list(range(10))
    assert list(generator()) == list(range(10))
    assert list(generator) == list(range(10))
    assert list(generator) == list(range(10))


def test_progressable_generator():
    def f_generator(limit):
        return (i for i in range(limit))

    generator = er_text.ProgressableGenerator(f_generator, limit=10)
    assert list(generator) == list(range(10))
    assert list(generator) == list(range(10))
    assert list(generator()) == list(range(10))
    assert list(generator()) == list(range(10))

    generator = er_text.ProgressableGenerator(f_generator, limit=10)
    assert list(generator) == list(range(10))
    assert list(generator) == list(range(10))
    assert list(generator()) == list(range(10))
    assert list(generator()) == list(range(10))


def test_split_into_words():
    elms = ["sentence one", "part 2."]
    got = er_text.split_into_words(elms)
    expected = ["sentence", "one", "part", "2."]
    assert list(got) == expected


def test_preprocess():
    texts = ["\n\nA\nb ", "C\r"]
    assert list(er_text.preprocess(texts)) == ["a b", "c"]
    assert list(er_text.preprocess(texts, to_lower=False)) == ["A b", "C"]
    assert (
        list(er_text.preprocess(texts, to_lower=False, remove_split_lines=False))
        == texts
    )

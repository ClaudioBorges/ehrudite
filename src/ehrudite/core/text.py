"""ehrpreper SentencePieceTokenizer"""

import ehrpreper
import sentencepiece as spm
import tqdm


class Generator:
    def __init__(self, raw_iterator):
        self._raw_iterator = raw_iterator

    def __call__(self):
        return self._raw_iterator


class RepeatableGenerator:
    def __init__(self, _func_generator, **kwargs):
        self._func_generator = _func_generator
        self._kwargs = kwargs
        self._generator = self._make_generator()

    def _make_generator(self):
        return self._func_generator(**self._kwargs)

    def __iter__(self):
        return self._make_generator()

    def __call__(self):
        return self._make_generator()


class LenghtableRepeatableGenerator(RepeatableGenerator):
    def __init__(self, _func_generator, _length=None, **kwargs):
        super(LenghtableRepeatableGenerator, self).__init__(_func_generator, **kwargs)
        self._length = _length if _length else sum([1 for _ in self._make_generator()])

    def __len__(self):
        return self._length


class ProgressableGenerator(RepeatableGenerator):
    def __init__(self, func_generator, n_items=None, **kwargs):
        self._last_progressable = None
        super(ProgressableGenerator, self).__init__(func_generator, **kwargs)

    def _make_generator(self):
        generator = super(ProgressableGenerator, self)._make_generator()
        progressable = tqdm.tqdm(iterable=generator, total=self._n_items)
        return progressable

    def __iter__(self):
        self._last_progressable = self._make_generator()
        return (item for item in self._last_progressable)

    def __next__(self):
        raise NotImplemented


def preprocess_text(text, to_lower=True, remove_split_lines=True):
    def _remove_split_lines(text):
        lines = text.splitlines()
        chunks = [line.strip() for line in lines if len(line) != 0]
        return " ".join(chunks)

    text_step_1 = text.lower() if to_lower else text
    text_step_2 = (
        _remove_split_lines(text_step_1) if remove_split_lines else text_step_1
    )
    return text_step_2


def preprocess_icds9(icds9, icd_10_limit=3, separator=" ", **kwargs):
    converter = ehrpreper.Icd9To10Converter()
    # Convert to ICD10, use the first 3 digits, remove duplications and sort
    codes = (preprocess_text(converter.convert(icd9)[:icd_10_limit]) for icd9 in icds9)
    deduped = list(set(codes))
    ordered = sorted(deduped)
    return separator.join(ordered)


def preprocess(texts, **kwargs):
    return (preprocess_text(text, **kwargs) for text in texts)


def split_into_words(texts):
    return (word for text in texts for word in text.split())

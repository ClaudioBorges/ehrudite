"""Ehrpreper statistic"""

from tqdm import tqdm
import ehrpreper
import ehrudite.core.text as er_text
import ehrudite.core.tokenizer.sentencepiece_tokenizer as sentencepiece
import ehrudite.core.tokenizer.wordpiece_tokenizer as wordpiece
import logging
import matplotlib.pyplot as plt
import numpy as np
import os


def _load_annotations_collections_from_ehrpreper(ehrpreper_file_name):
    logging.info(
        f"_load_annotations_collections_from_ehrpreper (file_name={ehrpreper_file_name})"
    )
    return (
        document.annotations
        for model in ehrpreper.load(ehrpreper_file_name)
        for document in model.documents
    )


def _load_contents_from_ehrpreper(ehrpreper_file_name):
    logging.info(f"_load_contents_from_ehrpreper (file_name={ehrpreper_file_name})")
    return (
        document.content
        for model in ehrpreper.load(ehrpreper_file_name)
        for document in model.documents
    )


def _from_tokenizer_ehpreper(
    ehrpreper_file_name, tokenizer, tokenize_and_get_num_tokens, output_path
):
    tokenizer_name = tokenizer.__class__.__name__
    logging.info(
        f"Generating tokenizer statistic (file_name={ehrpreper_file_name}, name={tokenizer_name})..."
    )

    n_contents = sum([1 for i in _load_contents_from_ehrpreper(ehrpreper_file_name)])

    n_sentences_per_content = []
    n_tokens_per_sentence = []
    n_tokens_per_content = []

    logging.info(f"Iterating over contents (n_contents={n_contents})...")
    for model in tqdm(iterable=ehrpreper.load(ehrpreper_file_name), total=n_contents):
        for document in model.documents:
            content = document.content
            n_sentences = 0
            n_tokens_acc = 0
            for sentence in er_text.texts_to_sentences([content]):
                n_sentences += 1
                n_tokens = tokenize_and_get_num_tokens(tokenizer, sentence)
                n_tokens_acc += n_tokens
                n_tokens_per_sentence.append(n_tokens)
            n_sentences_per_content.append(n_sentences)
            n_tokens_per_content.append(n_tokens_acc)
    logging.info(
        f"{tokenizer_name} statistics"
        f"\n\tsentences_per_contens"
        f"\n\t\tmean={np.mean(n_sentences_per_content)}"
        f"\n\t\tstd={np.std(n_sentences_per_content)}"
        f"\n\t\tmax={np.max(n_sentences_per_content)}"
        f"\n\t\tmin={np.min(n_sentences_per_content)}"
        f"\n\t\thistogram={np.histogram(n_sentences_per_content, density=True)}"
        f"\n\t\ttotal={len(n_sentences_per_content)}"
        f"\n\ttokens_per_content"
        f"\n\t\tmean={np.mean(n_tokens_per_content)}"
        f"\n\t\tstd={np.std(n_tokens_per_content)}"
        f"\n\t\tmax={np.max(n_tokens_per_content)}"
        f"\n\t\tmin={np.min(n_tokens_per_content)}"
        f"\n\t\thistogram={np.histogram(n_tokens_per_content, density=True)}"
        f"\n\t\ttotal={len(n_tokens_per_content)}"
        f"\n\ttokens_per_sentence"
        f"\n\t\tmean={np.mean(n_tokens_per_sentence)}"
        f"\n\t\tstd={np.std(n_tokens_per_sentence)}"
        f"\n\t\tmax={np.max(n_tokens_per_sentence)}"
        f"\n\t\tmin={np.min(n_tokens_per_sentence)}"
        f"\n\t\thistogram={np.histogram(n_tokens_per_sentence, density=True)}"
        f"\n\t\ttotal={len(n_tokens_per_sentence)}"
    )
    plt.figure(f"{tokenizer_name} Sentences' number per content")
    _ = plt.hist(n_sentences_per_content, bins="auto", density=True)
    plt.xlabel("Sentences' number per content")
    plt.ylabel("Density")
    plt.title(
        f"{tokenizer_name}\nProbability distribution of the sentences' number per content"
    )
    plt.savefig(
        os.path.join(output_path, f"{tokenizer_name}-sentences-per-content.png")
    )

    plt.figure(f"{tokenizer_name} Tokens' number per content")
    _ = plt.hist(n_tokens_per_content, bins="auto", density=True)
    plt.xlabel("Tokens' number per content")
    plt.ylabel("Density")
    plt.title(
        f"{tokenizer_name}\nProbability distribution of the tokens' number per content"
    )
    plt.savefig(os.path.join(output_path, f"{tokenizer_name}-tokens-per-content.png"))

    plt.figure(f"{tokenizer_name} Tokens' number per sentence")
    _ = plt.hist(n_tokens_per_sentence, bins=[i for i in range(80)], density=True)
    plt.xlabel("Tokens' number per sentence")
    plt.ylabel("Density")
    plt.title(
        f"{tokenizer_name}\nProbability distribution of the tokens' number per sentence"
    )
    plt.savefig(os.path.join(output_path, f"{tokenizer_name}-tokens-per-sentence.png"))


def from_wordpiece_ehrpreper(ehrpreper_file_name, wordpiece_vocab_file, output_path):
    def tokenize_and_get_num_tokens(tokenizer, sentence):
        tokens = tokenizer.tokenize(sentence)
        return tokens.flat_values.shape.num_elements()

    logging.info(f"Starting Wordpiece statistic...")
    tokenizer = wordpiece.WordpieceTokenizer(wordpiece_vocab_file)
    _from_tokenizer_ehpreper(
        ehrpreper_file_name,
        tokenizer,
        tokenize_and_get_num_tokens,
        output_path,
    )


def from_sentencepiece_ehrpreper(
    ehrpreper_file_name, sentencepiece_model_file, output_path
):
    def tokenize_and_get_num_tokens(tokenizer, sentence):
        tokens = tokenizer.tokenize(sentence)
        return tokens.shape.num_elements()

    logging.info(f"Starting Sentencepiece statistic...")
    tokenizer = sentencepiece.SentencepieceTokenizer(sentencepiece_model_file)
    _from_tokenizer_ehpreper(
        ehrpreper_file_name,
        tokenizer,
        tokenize_and_get_num_tokens,
        output_path,
    )


def from_ehrpreper(ehrpreper_file_name, output_path):
    def _annotations_stat(annotations_collections):
        n_annotations = [len(annotations) for annotations in annotations_collections]
        logging.info(
            f"Annotations' number"
            + f"\n\tmean={np.mean(n_annotations)}"
            + f"\n\tstd={np.std(n_annotations)}"
            + f"\n\ttotal={len(n_annotations)}"
            + f"\n\tmin={np.min(n_annotations)}"
            + f"\n\tmax={np.max(n_annotations)}"
            + f"\n\thistogram={np.histogram(n_annotations, density=True)}"
        )
        plt.figure("Annotations' number per content")
        _ = plt.hist(
            n_annotations, bins=np.max(n_annotations) + 1, rwidth=0.8, density=True
        )
        plt.xlabel("Annotations' number per content")
        plt.ylabel("Density")
        plt.title("Probability distribution of the annotations' number per content")
        plt.savefig(os.path.join(output_path, f"ehrpreper-annotations-per-content-png"))

    def _content_stat(contents):
        n_len_contents = [len(content) for content in contents]
        logging.info(
            f"Characters' number per content"
            + f"\n\tmean={np.mean(n_len_contents)}"
            + f"\n\tstd={np.std(n_len_contents)}"
            + f"\n\ttotal={len(n_len_contents)}"
            + f"\n\tmin={np.min(n_len_contents)}"
            + f"\n\tmax={np.max(n_len_contents)}"
            + f"\n\thistogram={np.histogram(n_len_contents, density=True)}"
        )
        plt.figure("Characters' number per content")
        _ = plt.hist(n_len_contents, bins="auto", density=True)
        plt.xlabel("Characters' number per content")
        plt.ylabel("Density")
        plt.title("Probability distribution of the characters' number per content")
        plt.savefig(os.path.join(output_path, f"ehrpreper-characters-per-content-png"))

    logging.info(f"Generating statistic (file_name={ehrpreper_file_name})...")

    annotations_collections = _load_annotations_collections_from_ehrpreper(
        ehrpreper_file_name
    )
    contents = _load_contents_from_ehrpreper(ehrpreper_file_name)

    _annotations_stat(annotations_collections)
    _content_stat(contents)

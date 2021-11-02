"""Ehrpreper statistic"""


from collections import Counter
from tqdm import tqdm
import ehrpreper
import ehrudite.core.pipeline as pip
import ehrudite.core.text as er_text
import ehrudite.core.tokenizer.sentencepiece as sentencepiece
import ehrudite.core.tokenizer.wordpiece as wordpiece
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
            for sentence in er_text.preprocess([content]):
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
    _ = plt.hist(n_tokens_per_sentence, bins="auto", density=True)
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

        # Fine-grain statistic
        k_folded = pip.ehrpreper_k_fold_gen(ehrpreper_file_name)
        for run_id, (
            train_xy,
            test_xy,
        ) in enumerate(k_folded):
            _, train_y = pip.unpack_2d(train_xy)
            _, test_y = pip.unpack_2d(test_xy)

            def make_counter(y):
                return Counter(
                    [
                        annotation
                        for annotations in y
                        for annotation in annotations.split()
                    ]
                )

            def make_stats(name, cnt):
                cnt_vals = list(cnt.values())
                logging.info(
                    f"Annotations' counter for {name} run_id={run_id}"
                    + f"\n\tmean={np.mean(cnt_vals)}"
                    + f"\n\tstd={np.std(cnt_vals)}"
                    + f"\n\ttotal={len(cnt_vals)}"
                    + f"\n\tmin={np.min(cnt_vals)}"
                    + f"\n\tmax={np.max(cnt_vals)}"
                    + f"\n\thistogram={np.histogram(cnt_vals, density=True)}"
                    + f"\n\ttop20={cnt.most_common(20)}"
                )

            cnt_train = make_counter(train_y)
            cnt_test = make_counter(test_y)
            make_stats("TRAIN", cnt_train)
            make_stats("TEST", cnt_test)

            unprecedent = [elm for elm in cnt_test if elm not in cnt_train]
            logging.info(
                f"Number of unique TEST annotation run_id={run_id}"
                + f"\n\tn_unprecedent={len(unprecedent)}"
                + f"\n\tunprecedent={set(unprecedent)}"
            )

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
    # _content_stat(contents)

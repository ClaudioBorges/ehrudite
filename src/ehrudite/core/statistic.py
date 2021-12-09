"""Ehrpreper statistic"""


from collections import Counter
from tqdm import tqdm
import ehrpreper
import ehrudite.core.pipeline as pip
import ehrudite.core.pipeline.tokenizer as pip_tok
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
    sequences,
    output_path,
):
    logging.info(f"Iterating over sequences (n_sequences={len(sequences)})...")
    n_tokens_per_sequence = [sequence.shape[0] for sequence in tqdm(sequences)]

    logging.info(
        f"Statistics"
        f"\n\ttokens_per_sequence"
        f"\n\t\tmean={np.mean(n_tokens_per_sequence)}"
        f"\n\t\tstd={np.std(n_tokens_per_sequence)}"
        f"\n\t\tmax={np.max(n_tokens_per_sequence)}"
        f"\n\t\tmin={np.min(n_tokens_per_sequence)}"
        f"\n\t\thistogram={np.histogram(n_tokens_per_sequence, density=True)}"
        f"\n\t\ttotal={len(n_tokens_per_sequence)}"
    )
    return n_tokens_per_sequence


def from_tokenizer_ehrpreper(ehrpreper_file, tokenizer_type, output_path):
    logging.info(f"Generating tokenizer statistic (name={tokenizer_type})...")

    n_tokens_per_sequence_x = []
    n_tokens_per_sequence_y = []

    k_folded = pip.ehrpreper_k_fold_gen(ehrpreper_file)
    for run_id, (
        train_xy,
        test_xy,
    ) in enumerate(k_folded):
        tok_x, tok_y = pip_tok.restore(run_id, tokenizer_type)
        train_x, train_y = pip.unpack_2d(train_xy)
        test_x, test_y = pip.unpack_2d(test_xy)

        logging.info(f"Contents...")
        x_sequences = er_text.LenghtableRepeatableGenerator(
            lambda: (tok_x.tokenize(x) for xs in [train_x, test_x] for x in xs),
            _length=len(train_x) + len(test_x),
        )
        n_tokens_per_sequence_x += _from_tokenizer_ehpreper(
            x_sequences, os.path.join(output_path, str(tokenizer_type), "contents")
        )

        logging.info(f"Annotations...")
        y_sequences = er_text.LenghtableRepeatableGenerator(
            lambda: (tok_y.tokenize(y) for ys in [train_y, test_y] for y in ys),
            _length=len(train_y) + len(test_y),
        )
        n_tokens_per_sequence_y += _from_tokenizer_ehpreper(
            y_sequences, os.path.join(output_path, str(tokenizer_type), "annotations")
        )

    logging.info(
        f"Contents statistic for {tokenizer_type}"
        f"\n\ttokens_per_sequence"
        f"\n\t\tmean={np.mean(n_tokens_per_sequence_x)}"
        f"\n\t\tstd={np.std(n_tokens_per_sequence_x)}"
        f"\n\t\tmax={np.max(n_tokens_per_sequence_x)}"
        f"\n\t\tmin={np.min(n_tokens_per_sequence_x)}"
        f"\n\t\thistogram={np.histogram(n_tokens_per_sequence_x, density=True)}"
        f"\n\t\ttotal={len(n_tokens_per_sequence_x)}"
        f"Annotations statistic for {tokenizer_type}"
        f"\n\ttokens_per_sequence"
        f"\n\t\tmean={np.mean(n_tokens_per_sequence_y)}"
        f"\n\t\tstd={np.std(n_tokens_per_sequence_y)}"
        f"\n\t\tmax={np.max(n_tokens_per_sequence_y)}"
        f"\n\t\tmin={np.min(n_tokens_per_sequence_y)}"
        f"\n\t\thistogram={np.histogram(n_tokens_per_sequence_y, density=True)}"
        f"\n\t\ttotal={len(n_tokens_per_sequence_y)}"
    )

    plt.figure(f"{tokenizer_type} - Tokens' number per content sequence")
    _ = plt.hist(n_tokens_per_sequence_x, bins="auto", density=True)
    plt.xlabel("Tokens' number per content sequence")
    plt.ylabel("Density")
    plt.title(f"Probability distribution of the tokens' number per content sequence")
    plt.savefig(os.path.join(output_path, f"{tokenizer_type}-tokens-per-content.png"))

    plt.figure(f"{tokenizer_type} - Tokens' number per annotation sequence")
    _ = plt.hist(n_tokens_per_sequence_y, bins="auto", density=True)
    plt.xlabel("Tokens' number per annotation sequence")
    plt.ylabel("Density")
    plt.title(f"Probability distribution of the tokens' number per annotation sequence")
    plt.savefig(
        os.path.join(output_path, f"{tokenizer_type}-tokens-per-annotation.png")
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

        annotations_set_train = []
        annotations_set_test = []
        annotations_set = []
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

            partial_annotations_set_train = set(cnt_train.elements())
            partial_annotations_set_test = set(cnt_test.elements())
            partial_annotations_set = partial_annotations_set_train.union(
                partial_annotations_set_test
            )

            logging.info(
                f"Partial annotations' distinct classes (train) run_id={run_id}"
                + f"\n\tlen={len(partial_annotations_set_train)}"
                + f"\nPartial Annotations' distinct classes (test) run_id={run_id}"
                + f"\n\tlen={len(partial_annotations_set_test)}"
                + f"\nPartial Annotations' distinct classes run_id={run_id}"
                + f"\n\tlen={len(partial_annotations_set)}"
            )
            annotations_set_train.append(partial_annotations_set_train)
            annotations_set_test.append(partial_annotations_set_test)
            annotations_set.append(partial_annotations_set)

        n_annotations_set_train = [len(vals) for vals in annotations_set_train]
        n_annotations_set_test = [len(vals) for vals in annotations_set_test]
        n_annotations_set = [len(vals) for vals in annotations_set]

        logging.info(
            f"Annotations' distinct classes (train)"
            + f"\n\tmean={np.mean(n_annotations_set_train)}"
            + f"\n\tstd={np.std(n_annotations_set_train)}"
            + f"\n\tmin={np.min(n_annotations_set_train)}"
            + f"\n\tmax={np.max(n_annotations_set_train)}"
            + f"\n\tvalues={n_annotations_set_train}"
            + f"\nAnnotations' distinct classes (test)"
            + f"\n\tmean={np.mean(n_annotations_set_test)}"
            + f"\n\tstd={np.std(n_annotations_set_test)}"
            + f"\n\tmin={np.min(n_annotations_set_test)}"
            + f"\n\tmax={np.max(n_annotations_set_test)}"
            + f"\n\tvalues={n_annotations_set_test}"
            + f"\nAnnotations' distinct classes"
            + f"\n\tmean={np.mean(n_annotations_set)}"
            + f"\n\tstd={np.std(n_annotations_set)}"
            + f"\n\tmin={np.min(n_annotations_set)}"
            + f"\n\tmax={np.max(n_annotations_set)}"
            + f"\n\tvalues={n_annotations_set}"
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
    _content_stat(contents)


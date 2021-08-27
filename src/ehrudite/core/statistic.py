"""Ehrpreper statistic"""

import ehrpreper
import logging
import matplotlib.pyplot as plt
import numpy as np


def from_ehrpreper(ehrpreper_file_name, include_graphs=True):
    def _annotations_stat(annotations_list, include_graphs):
        n_annotations = [len(annotations) for annotations in annotations_list]
        logging.info(
            f"Annotations' number"
            + f"\n\tmean={np.mean(n_annotations)}"
            + f"\n\tstd={np.std(n_annotations)}"
            + f"\n\ttotal={len(n_annotations)}"
            + f"\n\tmin={np.min(n_annotations)}"
            + f"\n\tmax={np.max(n_annotations)}"
            + f"\n\thistogram={np.histogram(n_annotations, density=True)}"
        )
        if include_graphs:
            plt.figure("Annotations' number per content")
            _ = plt.hist(
                n_annotations, bins=np.max(n_annotations) + 1, rwidth=0.8, density=True
            )
            plt.xlabel("Annotations' number per content")
            plt.ylabel("Density")
            plt.title("Probability distribution of the annotations' number per content")
            plt.show(block=False)

    def _content_stat(contents, include_graphs):
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
        if include_graphs:
            plt.figure("Characters' number per content")
            _ = plt.hist(n_len_contents, bins="auto", density=True)
            plt.xlabel("Characters' number per content")
            plt.ylabel("Density")
            plt.title("Probability distribution of the characters' number per content")
            plt.show(block=False)

    logging.info(f"Generating statistic (file_name={ehrpreper_file_name})...")

    annotations_list = []
    contents = []
    for model in ehrpreper.load(ehrpreper_file_name):
        for document in model.documents:
            annotations_list.append(document.annotations)
            contents.append(document.content)

    _annotations_stat(annotations_list, include_graphs)
    _content_stat(contents, include_graphs)
    if include_graphs:
        plt.show()

"""Ehrpreper statistic"""

import ehrpreper
import logging
import matplotlib.pyplot as plt
import numpy as np


def generate(ehrpreper_file_name, include_graphs=True):
    def _annotations_stat(annotations_list, include_graphs):
        n_annotations= [len(annotations) for annotations in annotations_list]
        print(f'Annotations\' quantity mean={np.mean(n_annotations)}'
              + f', std={np.std(n_annotations)}, total={len(n_annotations)}'
              + f', min={np.min(n_annotations)}, max={np.max(n_annotations)}'
              + f', histogram={np.histogram(n_annotations, density=True)}')
        if include_graphs:
            _ = plt.hist(n_annotations, bins=np.max(n_annotations)+1, rwidth=0.8, density=True)
            plt.xlabel('Quantity of annotations per content')
            plt.ylabel('Density')
            plt.title('Probability distribution of the annotations\' quantity per content')
            plt.show(block=False)

    def _content_stat(contents, include_graphs):
        n_len_contents= [len(content) for content in contents]
        print(f'Content\'s length mean={np.mean(n_len_contents)}'
              + f', std={np.std(n_len_contents)}, total={len(n_len_contents)}'
              + f', min={np.min(n_len_contents)}, max={np.max(n_len_contents)}'
              + f', histogram={np.histogram(n_len_contents, density=True)}')
        if include_graphs:
            _ = plt.hist(n_len_contents, bins='auto', density=True)
            plt.xlabel('Length of contens')
            plt.ylabel('Density')
            plt.title('Probability distribution of the content\'s length')
            plt.show(block=False)

    logging.info(f'Generating statistic (file_name={ehrpreper_file_name})...')

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


generate("output.xml")

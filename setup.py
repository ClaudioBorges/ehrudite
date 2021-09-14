#!/usr/bin/env python

"""The script for setting up ehrudite."""


import sys

if sys.version_info[0] < 3:
    raise Exception("Ehrudite does not support Python 2. Please upgrade to Python 3.")

import configparser
from os.path import dirname
from os.path import join

from setuptools import find_packages, setup

# Get the global config info as currently stated
# (we use the config file to avoid actually loading any python here)
config = configparser.ConfigParser()
config.read(["src/ehrudite/config.ini"])
version = config.get("ehrudite", "version")


def read(*names, **kwargs):
    """Read a file and return the contents as a string."""
    return open(
        join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ).read()


setup(
    name="ehrudite",
    version=version,
    license="Apache License 2.0",
    description="An deep learning (DL) framework for electronic health recods (EHR)",
    long_description=read("README.md"),
    # Make sure pypi is expecting markdown!
    long_description_content_type="text/markdown",
    author="Claudio Borges",
    author_email="claudio.borges.jr@gmail.com",
    python_requires=">=3.6",
    keywords=["ehrudite", "ehr", "dl", "ml"],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "configparser",
        "ehrpreper @ git+https://github.com/ClaudioBorges/ehrpreper.git",
        "matplotlib",
        "numpy",
        "sentencepiece == 0.1.*",
        "tensorflow-text",
        "tensorflow",
        "tqdm",
        "gensim < 4.1",
        "python-Levenshtein",
        "scikit-learn",
    ],
    entry_points={
        "console_scripts": [
            "ehrudite = ehrudite.cli:cli",
            "ehrudite-stat = ehrudite.cli.statistic:cli",
            "ehrudite-tok = ehrudite.cli.tokenizer:cli",
        ],
    },
)

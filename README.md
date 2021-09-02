# Introduction

EHRudite is an deep learning (DL) framework for eletronic health records (EHR).

# Contributing
## Preparation

Make sure to copy the mimic-III v1.4 `NOTEEVENTS.csv.gz` and `DIAGNOSES_ICD.csv.gz` to the `input-data/mimicIII`.

## Testing
### Docker

Build the docker image executing:
```
docker build ehrudite -t ehrudite
```

and then run it:
```
docker run -it ehrudite
```

### Local
To test locally, Ehrudite uses `tox`, which means you can build locally using:
```
pip install tox
tox
```
#### Using your local version

To trial using your local development branch of Ehrudite, I recommend you use
a virtual environment. e.g:

```shell
python3 -m venv .venv
source .venv/bin/activate
```
> `python3 -m venv .venv` creates a new virtual environment (in current working
> directory) called `.venv`.
> `source .venv/bin/activate` activates the virtual environment so that packages
> can be installed/uninstalled into it. [More info on venv](https://docs.python.org/3/library/venv.html).

Once you're in a virtual environment, run:

```shell
pip install -Ur requirements.txt -Ur requirements_dev.txt
python setup.py develop
```

> `setup.py develop` installs the package using a link to the source code so
> that any changes which you make will immediately be available for use.
>
> `pip install -Ur requirements.txt -Ur requirements_dev.txt` installs the
> project dependencies as well as the dependencies needed to run linting,
> formatting, and testing commands. This will install the most up-to-date
> package versions for all dependencies.

## CLIs

### Statistic

Show dataset statistic:
```
ehrudite-stat ../data/ehpreper.xml -g -v
```

Show dataset and tokenizer statistic:
```
ehrudite-stat ../data/ehrpreper.xml -vv -w ../data/tok/wordpiece/wordpiece.vocab -s ../data/tok/sentencepiece/sentencepiece.model -o ../data/statistics/
```

### Tokenizer

Tokenize using wordpiece:
```
ehrudite-tok -g -e ../data/ehrpreper.xml -vv ../data/tok/wordpiece/ wordpiece
```

Tokenizer using sentencepiece:
```
ehrudite-tok -g -e ../data/ehrpreper.xml -vv ../data/tok/sentencepiece/ sentencepiece
```

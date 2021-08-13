# Introduction

EHRudite is an deep learning (DL) framework for eletronic health records (EHR).

# Contributing
## Preparation

Make sure to copy the mimic-III v1.4 `NOTEEVENTS.csv.gz` and `DIAGNOSES_ICD.csv.gz` to the `input-data/mimicIII`.

## Testing

Build the docker image executing:
```
docker build ehrudite -t ehrudite
```

and then run it:
```
docker run -it ehrudite
```

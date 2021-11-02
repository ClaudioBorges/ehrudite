"""init py for pipeline."""

import ehrpreper
import ehrudite.core.text as er_text
import logging
import sklearn.model_selection as skl_msel
import tqdm

BASE_PATH = "../ehr-data/"


def make_progressable(iterable):
    return (i for i in tqdm.tqdm(iterable=iterable))


def unpack_2d(packed):
    def _extract(pos):
        return (elm[pos] for elm in packed)

    return (
        er_text.LenghtableRepeatableGenerator(_extract, _length=len(packed), pos=0),
        er_text.LenghtableRepeatableGenerator(_extract, _length=len(packed), pos=1),
    )


def ehrpreper_k_fold_gen(ehrpreper_file, n_splits=4):
    # Data builder from ehrpreper documents to tuple of X and Y
    def data_builder():
        documents = ehrpreper.document_entity_generator(ehrpreper_file)
        return (
            (
                er_text.preprocess_text(document.content),
                er_text.preprocess_icds9(document.annotations),
            )
            for document in documents
        )

    # Build the Generator
    data = er_text.LenghtableRepeatableGenerator(data_builder)

    logging.info(f"ehrpre_k_fold_gen (n_splits={n_splits}, data_len={len(data)})")
    kf = skl_msel.KFold(n_splits=n_splits)
    # Make indexes to be sure data can be a generator
    source_idxs = [i for i in range(len(data))]
    for i, (
        train_idxs,
        test_idxs,
    ) in enumerate(kf.split(source_idxs)):
        logging.debug(
            f"Generating k-fold"
            + f"(index={i}, len(train)={len(train_idxs)}, len(test)={len(test_idxs)})"
        )

        def k_fold_builder(data, idxs):
            return (elm for idx, elm in enumerate(data) if idx in idxs)

        train = er_text.LenghtableRepeatableGenerator(
            k_fold_builder, _length=len(train_idxs), data=data, idxs=set(train_idxs)
        )
        test = er_text.LenghtableRepeatableGenerator(
            k_fold_builder, _length=len(test_idxs), data=data, idxs=set(test_idxs)
        )

        # Generate the tuples of train and test
        yield (
            train,
            test,
        )

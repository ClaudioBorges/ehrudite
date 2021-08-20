from apache_beam.io.filesystems import FileSystems as beam_fs
from apache_beam.options.pipeline_options import PipelineOptions
from typing import Iterable, Type, List
import apache_beam as beam

"""ehrpreper SentencePieceTokenizer"""


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.pvalue.PBegin)
@beam.typehints.with_output_types(Type[ehrpreper.core.ModelEntity])
def _read_ehrpreper_files(
    pbegin: beam.pvalue.PBegin, file_patterns: List[str]
) -> beam.PCollection[Type[ehrpreper.core.ModelEntity]]:
    def expand_pattern(pattern: str) -> Iterable[str]:
        for match_result in beam_fs.match([pattern])[0].metadata_list:
            yield match_result.path

    def read_ehrpreper_lines(
        file_name: str,
    ) -> Iterable[Type[ehrpreper.core.ModelEntity]]:
        for model in ehrpreper.load(file_name):
            yield model

    return (
        pbegin
        | "Create file patterns" >> beam.Create(file_patterns)
        | "Expand file patterns" >> beam.FlatMap(expand_pattern)
        | "Read ehrpreper lines" >> beam.FlatMap(read_ehrpreper_lines)
    )


def start(input_patterns=[]):
    options = PipelineOptions(flags=[], type_check_additional="all")
    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline
            | "Read ehrpreper files" >> _read_ehrpreper_files(input_patterns)
            | "Print elements" >> beam.Map(print)
        )

import sys

from data.graphones import (
    ALIGNMENT_FILE,
    align_corpus,
    build_units,
    espeak,
    type_counts,
    word_parts,
    write_tables,
)
from framework.storage import write_json
from workers.entry import invocation


def main(arguments: list[str]) -> None:
    run = invocation(arguments)
    corpus = run.corpus()
    parts = word_parts(corpus)
    counts = type_counts(corpus)
    alignments = align_corpus(corpus, parts, counts, run.config.graphones, espeak)
    root = run.layout.graphones
    root.mkdir(parents=True, exist_ok=True)
    write_json(
        root / ALIGNMENT_FILE, dict(zip(corpus.languages, alignments, strict=True))
    )
    units = build_units(parts, alignments, corpus.max_bytes)
    write_tables(root, units, counts, run.config.code_widths(), run.config.root_seed)


if __name__ == "__main__":
    main(sys.argv[1:])

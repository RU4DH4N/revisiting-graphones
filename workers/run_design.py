import sys

from framework.storage import write_json
from workers import design
from workers.entry import invocation


def main(arguments: list[str]) -> None:
    run = invocation(arguments)
    corpus = run.corpus()
    words = corpus.train_words()
    found = design.sized(
        run.config, corpus.max_bytes, words, design.steps(run.config, words)
    )
    write_json(run.layout.design, design.sizing_record(found))


if __name__ == "__main__":
    main(sys.argv[1:])

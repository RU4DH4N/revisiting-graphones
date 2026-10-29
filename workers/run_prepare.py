import sys

from data.benchmarks import fetch_benchmarks
from data.bilingual import build, fineweb_sources
from workers.entry import invocation


def main(arguments: list[str]) -> None:
    run = invocation(arguments)
    data = run.config.data
    benchmarks = fetch_benchmarks(data.benchmarks, data.languages)
    build(data, run.layout.corpus, fineweb_sources(data), benchmarks)


if __name__ == "__main__":
    main(sys.argv[1:])

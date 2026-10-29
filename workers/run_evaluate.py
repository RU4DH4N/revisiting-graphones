import sys
from dataclasses import asdict

import numpy as np

from data.bilingual import BilingualCorpus
from data.graphones import read_tables
from framework.seeds import seed_for
from framework.storage import write_json
from workers.arms import load_model
from workers.batches import Expander
from workers.benchmarks import Measures, benchmark_scores, evaluation_record
from workers.entry import SeedInvocation, seed_invocation
from workers.evaluate import Scorer, Scores


def scorer_for(task: SeedInvocation, corpus: BilingualCorpus) -> Scorer:
    run = task.run
    tables = read_tables(run.layout.graphones)
    model = load_model(
        task.checkpoint(),
        run.config,
        run.sizing(),
        tables,
        task.arm,
        task.replicate,
    )
    return Scorer(
        model.to(run.device),
        Expander(corpus, tables.codes, tables.lengths, run.device),
        run.config.context_words,
        run.device,
        run.config.precision,
        seed_for(run.config.root_seed, "evaluation-order"),
    )


def save_scores(task: SeedInvocation, found: Scores) -> None:
    scores = task.scores()
    scores.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        scores,
        types=found.types,
        languages=found.languages,
        nats=found.nats,
        symbols=found.symbols,
    )


def main(arguments: list[str]) -> None:
    task = seed_invocation(arguments)
    run = task.run
    corpus = run.corpus()
    scorer = scorer_for(task, corpus)
    evaluation = run.config.evaluation
    measures = Measures(
        evaluation.probe_steps,
        evaluation.probe_decay,
        evaluation.neighbours,
        seed_for(run.config.root_seed, "control-tags"),
    )
    languages = run.config.data.languages
    ids = corpus.language_ids(languages)
    found = scorer.score(corpus.parts[task.split], ids)
    save_scores(task, found)
    record = evaluation_record(
        found.bits_per_byte(np.ones(len(found.nats), dtype=bool)),
        {
            language: found.bits_per_byte(found.languages == k)
            for language, k in zip(languages, ids, strict=True)
        },
        len(found.nats),
        benchmark_scores(scorer, corpus, task.split, ids, measures),
    )
    write_json(task.evaluation(), asdict(record))


if __name__ == "__main__":
    main(sys.argv[1:])

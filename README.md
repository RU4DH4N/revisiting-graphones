# Revisiting Graphones for Language Modelling

Supplementary work to [Revisiting Graphones for Language Modelling](10.5281/zenodo.24356108).

## Layout

| Directory | Contents |
|---|---|
| `experiments/` | One entry point per experiment |
| `pipeline/` | Graph steps: prepare, design, graphones, mix tables, train, release steps, evaluate, experiment |
| `workers/` | Workers the steps call, and the analysis that compares arms |
| `models/` | Graphone encoders, mixing gates, word trunk and receivers |
| `data/` | Corpus building, letter-sound alignment and graphone codes |
| `orchestrator/` | Dependency graph, caching, plan export |
| `framework/` | Checkpoint, runtime, seed, git revision and source-fingerprint utilities |
| `figures/` | Paper figures, LaTeX table bodies and `numbers.tex`, drawn from finished results |
| `config/` | `paper.json` holds the shared settings; each language pair, `en.json` (English alone) and `smoke.json` extend it |
| `unit-tests/` | `uv run pytest` |

## Setup

```sh
brew install espeak-ng uv    # or your package manager of choice
uv run pytest
```

run the following commands below using `uv run` e.g. `uv run -m orchestrator experiments/learned_mix.py ...`
espeak-ng must be 1.52.

## Data

| Part | Source |
|---|---|
| English | [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) `sample/10BT`, pinned revision, ODC-By |
| French, Russian, Spanish, Portuguese | [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) `fra_Latn`, `rus_Cyrl`, `spa_Latn`, `por_Latn`, pinned revision, ODC-By |
| Pronunciation | [espeak-ng](https://github.com/espeak-ng/espeak-ng) 1.52 through [phonemizer](https://github.com/bootphon/phonemizer) |
| Parallel | [FLORES-200](https://github.com/facebookresearch/flores) devtest, pinned hash, CC BY-SA 4.0 |
| Part of speech | [Universal Dependencies](https://universaldependencies.org) r2.18 dev and test: EWT, French/Russian/Spanish/Portuguese GSD, CC BY-SA 4.0 |
| Grammaticality | [MultiBLiMP 1.0](https://huggingface.co/datasets/jumelet/multiblimp), pinned revision, CC BY 4.0 |

`prepare` streams each language and assigns documents to `train`, `selection` or `reporting` by a hash of their
text. `graphones` phonemises every word and aligns letters to sounds with no prior letter-sound knowledge: the
alignment is learned from the words alone, and a two-letter spelling of one sound (sh, qu, an) gives that sound to
its first letter. It then turns every word into letter, sound and joint units with fixed random codes.

## Experiments

| Experiment | Result | Entry point |
|---|---|---|
| E1. | TBD | `experiments/projection_absorbs.py` |
| E2. | TBD | `experiments/learned_mix.py` |
| E3. | TBD | `experiments/mix_ablation.py` |
| E4. | TBD | `experiments/fixed_tables.py` |
| E5. | TBD | `experiments/receiver_size.py` |
| E6. | TBD | `experiments/headline.py` |
| E7. | TBD | `experiments/graphones_vs_letters.py` |
| E8. | TBD | `experiments/release_timing.py` |
| E9. | TBD | `experiments/trajectory.py` |
| E10. | TBD | `experiments/letter_cut.py` |
| E11. | TBD | `experiments/graphones_vs_bytes.py` |
| E12. | TBD | `experiments/rotation_matches_projection.py` |
| E13. | TBD | `experiments/rotation_gain.py` |
| E14. | TBD | `experiments/one_language.py` |
| E15. | TBD | `experiments/held_rotation.py` |
| E16. | TBD | `experiments/projection_control.py` |

Every experiment compares its arms on bits per byte, MultiBLiMP accuracy, a linear part-of-speech probe (accuracy,
selectivity, zero-shot transfer), FLORES retrieval (P@1, CSLS), linear CKA and late language share, each on the
split's half of the benchmark. `experiments/everything.py` runs all experiments on a language pair. E14 to E16 run
on English alone (`config/en.json`), so they have no retrieval, transfer, CKA or language share.

## Running an experiment

```sh
uv run -m orchestrator experiments/learned_mix.py --config config/smoke.json --split selection --device cpu --plan
uv run -m orchestrator experiments/learned_mix.py --config config/smoke.json --split selection --device cpu --run
```

`--plan` shows what is cached and what would run; `--run` runs the rest locally. Results are written to
`artefacts/<config>/experiments/<split>/`. Evaluate on `--split reporting` once, at the end.

To run elsewhere, export a plan per pair and merge them. The plan lists every task's command, dependencies and
outputs, and the git revision it came from. A cuda plan is refused unless the repository is committed:

```sh
for pair in en-fr en-ru es-pt; do
  uv run -m orchestrator experiments/everything.py --config config/$pair.json --split reporting --device cuda --export runs/$pair-plan.json
done
uv run -m orchestrator.merge runs/en-fr-plan.json runs/en-ru-plan.json runs/es-pt-plan.json --output runs/paper-plan.json
```

Figures and tables read finished results and never run anything; a missing input prints the command that makes it:

```sh
uv run -m figures all --config config/en-fr.json --split reporting
```

They are written to `artefacts/<config>/figures/<split>/`.

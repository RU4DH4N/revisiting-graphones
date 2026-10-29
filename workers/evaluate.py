import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from data.bilingual import Encoded
from framework.runtime import autocast_context
from models.bilingual import GraphoneModel
from workers.batches import Expander, Rows, evaluation_rows

EVALUATION_BATCH = 64


@dataclass(frozen=True)
class Scores:
    types: np.ndarray
    languages: np.ndarray
    nats: np.ndarray
    symbols: np.ndarray
    documents: np.ndarray

    def bits_per_byte(self, rows: np.ndarray) -> float:
        spelled = (self.symbols[rows] - 1).sum()
        return float(self.nats[rows].sum() / math.log(2) / max(spelled, 1))


@dataclass(frozen=True)
class Scorer:
    model: GraphoneModel
    expander: Expander
    events: int
    device: torch.device
    precision: str
    order_seed: int

    def score(self, part: Encoded, languages: list[int]) -> Scores:
        kept: list[tuple[Tensor, ...]] = []
        model = self.model.eval()
        rows = evaluation_rows(
            part, languages, self.events, EVALUATION_BATCH, self.order_seed
        )
        with torch.no_grad(), autocast_context(self.device, self.precision):
            for chunk in rows:
                batch = self.expander(chunk)
                nats = model.word_nll(
                    batch.inputs, batch.targets, model.messages(batch.context)
                )
                kept.append(
                    (
                        batch.scored,
                        batch.types,
                        batch.languages,
                        nats.float(),
                        batch.spelled,
                        batch.documents,
                    )
                )
        live, *columns = (
            torch.cat(column).cpu().numpy() for column in zip(*kept, strict=True)
        )
        return Scores(*(column[live] for column in columns))

    def represent(self, part: Encoded, documents: np.ndarray) -> list[np.ndarray]:
        found = []
        model = self.model.eval()
        with torch.no_grad(), autocast_context(self.device, self.precision):
            for rows in sentence_rows(part, documents, self.events):
                batch = self.expander(rows)
                found.append(model.trunk(batch.context).float().cpu().numpy())
        stacked = np.concatenate(found)
        lengths = np.minimum(np.diff(part.offsets)[documents], self.events)
        return [stacked[row, :length] for row, length in enumerate(lengths)]


def sentence_rows(part: Encoded, documents: np.ndarray, events: int) -> list[Rows]:
    rows = []
    for begin in range(0, len(documents), EVALUATION_BATCH):
        chunk = documents[begin : begin + EVALUATION_BATCH]
        words = np.zeros((len(chunk), events + 1), dtype=np.int64)
        for row, document in enumerate(chunk):
            start, stop = part.offsets[document], part.offsets[document + 1]
            found = np.asarray(part.ids[start : min(stop, start + events)])
            words[row, : len(found)] = found
        rows.append(
            Rows(
                words=torch.from_numpy(words),
                languages=torch.from_numpy(part.languages[chunk].astype(np.int64)),
                scored=torch.zeros(len(chunk), events, dtype=torch.bool),
                documents=torch.from_numpy(chunk.astype(np.int64)),
            )
        )
    return rows

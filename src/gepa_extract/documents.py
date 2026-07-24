"""Documents and training examples.

Documents are held as *references*, never as bytes on the instance. GEPA
pickles its optimisation state and JSON-dumps best outputs to the run
directory, so a `DataInst` carrying tens of megabytes of PDF would be
serialised repeatedly across a run. Bytes are read at call time and discarded.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["Document", "Example", "load_examples"]


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    path: Path
    media_type: str = "application/pdf"

    def read_bytes(self) -> bytes:
        return self.path.read_bytes()

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True, slots=True)
class Example:
    """One training/validation instance: a document and its gold extraction.

    This is the ``DataInst`` GEPA passes around. GEPA never inspects it.
    """

    document: Document
    gold: dict[str, Any]

    @property
    def doc_id(self) -> str:
        return self.document.doc_id


def load_examples(manifest_path: str | Path) -> list[Example]:
    """Load examples from a JSON manifest.

    The manifest is a list of ``{"doc_id", "file", "gold"}`` records, where
    ``file`` is resolved relative to the manifest's own directory so a dataset
    folder can be moved without editing it.
    """
    manifest_path = Path(manifest_path)
    records = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    return [
        Example(
            document=Document(
                doc_id=record["doc_id"],
                path=(base / record["file"]).resolve(),
                media_type=record.get("media_type", "application/pdf"),
            ),
            gold=record["gold"],
        )
        for record in records
    ]


def split(examples: list[Example], *, holdout: int) -> tuple[list[Example], list[Example]]:
    """Split off a holdout set from the end of ``examples``.

    Deliberately not shuffled: for extraction, the split that matters is by
    *vendor/layout*, and callers should order the list so that unseen layouts
    fall at the end. Random splits overstate accuracy, because a memorised
    layout cue scores well on a held-out document of the same template.
    """
    if holdout <= 0 or holdout >= len(examples):
        raise ValueError(f"holdout must be between 1 and {len(examples) - 1}")
    return examples[:-holdout], examples[-holdout:]


def iter_ids(examples: list[Example]) -> Iterator[str]:
    return (e.doc_id for e in examples)

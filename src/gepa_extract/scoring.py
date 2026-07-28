"""Per-field scoring, and the classification that makes feedback actionable.

Each field is scored independently, and the per-field scores are what GEPA
sees as objectives -- so a candidate that is best-in-class on ``total`` stays
on the Pareto front even when its mean is unremarkable.

The most valuable thing computed here is sibling detection. When ``total``
comes back holding the subtotal, the extracted value is not merely wrong: it
is *exactly some other field's gold value*. That is detectable without any
document coordinates, and it converts an opaque miss into "took the value
belonging to ``subtotal``" -- which tells the reflection model it is facing a
location problem, not a comprehension problem.

Rows of a repeating table are matched by *content*, not by position. An
extractor that emits line items in a different order, or drops one row, would
otherwise misalign every row after the first, and the resulting feedback is
worse than useless: comparing row 2 against row 1's gold routinely trips
sibling detection, so the reflection model is told it took a neighbouring
column when in truth it read the right cell of the wrong row, and it writes
positional anchors into a description that was already correct.

Alignment is computed once per array from *all* of that array's columns
jointly, then shared by each column's scoring. Doing it per column would let
``quantity`` and ``description`` choose different row orders, which destroys
the same-row scoping that makes sibling detection meaningful. Order itself is
never penalised: no description of ``line_items[].quantity`` can control the
order rows come back in, so scoring it would feed the optimiser noise it
cannot act on.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from gepa_extract.errors import ErrorClass
from gepa_extract.schema import ExtractionSchema
from gepa_extract.values import Match, compare, is_absent

__all__ = ["FORMAT_CREDIT", "DocumentOutcome", "FieldOutcome", "score_document"]

# Partial credit for "right value, wrong format". Not zero, because the model
# located the value correctly and that progress should survive the acceptance
# gate; not one, because a downstream consumer parsing the field would break.
FORMAT_CREDIT = 0.5

# Weights used only to decide which extracted row corresponds to which gold row.
# A normalised match identifies a row just as well as an exact one -- formatting
# has nothing to do with *which* row this is -- but scoring it fractionally
# breaks ties towards the exactly-matching row, which is the right bias.
_ALIGN_EXACT = 1.0
_ALIGN_NORMALIZED = 0.9


@dataclass(frozen=True, slots=True)
class FieldOutcome:
    path: str
    score: float
    error: ErrorClass
    extracted: Any
    gold: Any
    detail: str = ""
    # For SIBLING_VALUE: the field whose gold value was returned instead.
    # Structural rather than parsed back out of `detail`, because cross-document
    # pattern aggregation depends on it.
    related_path: str | None = None

    @property
    def is_failure(self) -> bool:
        return self.error.is_failure


@dataclass(frozen=True, slots=True)
class DocumentOutcome:
    doc_id: str
    fields: dict[str, FieldOutcome] = field(default_factory=dict)
    extraction_error: str | None = None

    @property
    def score(self) -> float:
        if not self.fields:
            return 0.0
        return sum(f.score for f in self.fields.values()) / len(self.fields)

    @property
    def objective_scores(self) -> dict[str, float]:
        """Per-field scores, in the shape GEPA's multi-objective front expects."""
        return {path: outcome.score for path, outcome in self.fields.items()}

    def failures(self) -> list[FieldOutcome]:
        return [f for f in self.fields.values() if f.is_failure]


def score_document(
    schema: ExtractionSchema,
    doc_id: str,
    extracted: Any,
    gold: Any,
    *,
    extraction_error: str | None = None,
    format_credit: float = FORMAT_CREDIT,
    array_order: str = "aligned",
) -> DocumentOutcome:
    """Score one extraction against its gold record, field by field.

    Args:
        array_order: How rows of a repeating table are paired with gold rows.
            ``"aligned"`` matches them by content, so a reordered or dropped row
            costs only itself. ``"positional"`` pairs row *i* with gold row *i*,
            which is right only when the extractor is required to preserve
            document order and a downstream consumer depends on it.
    """
    if array_order not in ("aligned", "positional"):
        raise ValueError(f"array_order must be 'aligned' or 'positional', got {array_order!r}.")
    gold_flat = schema.flatten(gold)

    if extraction_error is not None:
        return DocumentOutcome(
            doc_id=doc_id,
            extraction_error=extraction_error,
            fields={
                path: FieldOutcome(
                    path=path,
                    score=0.0,
                    error=ErrorClass.MISSING,
                    extracted=None,
                    gold=gold_flat.get(path),
                    detail=f"extraction failed for this document: {extraction_error}",
                )
                for path in schema.field_paths
            },
        )

    extracted_flat = schema.flatten(extracted)
    alignments = _align_arrays(schema, extracted_flat, gold_flat, format_credit) if array_order == "aligned" else {}

    outcomes: dict[str, FieldOutcome] = {}
    for spec in schema.fields:
        group = _array_group(spec.path) if spec.in_array else None
        outcomes[spec.path] = _classify(
            path=spec.path,
            extracted=extracted_flat.get(spec.path),
            gold=gold_flat.get(spec.path),
            gold_flat=gold_flat,
            in_array=spec.in_array,
            alignment=alignments.get(group) if group else None,
            format_credit=format_credit,
        )
    return DocumentOutcome(doc_id=doc_id, fields=outcomes)


def _classify(
    *,
    path: str,
    extracted: Any,
    gold: Any,
    gold_flat: dict[str, Any],
    in_array: bool,
    alignment: _RowAlignment | None,
    format_credit: float,
) -> FieldOutcome:
    if in_array:
        return _classify_array(
            path=path,
            extracted=extracted,
            gold=gold,
            gold_flat=gold_flat,
            alignment=alignment,
            format_credit=format_credit,
        )
    score, error, detail, related = _classify_scalar(
        path=path, extracted=extracted, gold=gold, gold_flat=gold_flat, format_credit=format_credit
    )
    return FieldOutcome(
        path=path, score=score, error=error, extracted=extracted, gold=gold, detail=detail, related_path=related
    )


def _classify_scalar(
    *,
    path: str,
    extracted: Any,
    gold: Any,
    gold_flat: dict[str, Any],
    format_credit: float,
) -> tuple[float, ErrorClass, str, str | None]:
    match compare(extracted, gold):
        case Match.EXACT:
            return 1.0, ErrorClass.CORRECT, "", None
        case Match.NORMALIZED:
            if is_absent(extracted) and is_absent(gold):
                return 1.0, ErrorClass.CORRECT, "", None
            return (
                format_credit,
                ErrorClass.FORMAT_MISMATCH,
                f"value is correct but formatted as {extracted!r} instead of {gold!r}",
                None,
            )

    if is_absent(gold):
        return (
            0.0,
            ErrorClass.HALLUCINATED,
            f"field is absent from this document, but {extracted!r} was returned",
            None,
        )
    if is_absent(extracted):
        return (
            0.0,
            ErrorClass.MISSING,
            f"field is present in this document with value {gold!r}, but null was returned",
            None,
        )

    sibling = _find_sibling(path, extracted, gold_flat)
    if sibling is not None:
        other_path, _ = sibling
        return (
            0.0,
            ErrorClass.SIBLING_VALUE,
            (
                f"returned {extracted!r}, which is the gold value of the '{other_path}' field, "
                f"instead of this field's value {gold!r}"
            ),
            other_path,
        )
    return 0.0, ErrorClass.WRONG_VALUE, f"returned {extracted!r}, expected {gold!r}", None


def _find_sibling(path: str, extracted: Any, gold_flat: dict[str, Any]) -> tuple[str, Any] | None:
    """Find another field whose gold value the extraction actually returned."""
    for other_path, other_gold in gold_flat.items():
        if other_path == path or is_absent(other_gold):
            continue
        candidates = other_gold if isinstance(other_gold, list) else [other_gold]
        for candidate in candidates:
            if is_absent(candidate):
                continue
            if compare(extracted, candidate) is not Match.DIFFERENT:
                return other_path, candidate
    return None


@dataclass(frozen=True, slots=True)
class _RowAlignment:
    """Which extracted row of a repeating table corresponds to which gold row.

    Keyed by gold row index so that iteration -- and the ``item N`` labels in
    feedback -- follow document order rather than the order the extractor
    happened to emit.
    """

    pairs: dict[int, int]  # gold row index -> extracted row index
    n_extracted: int
    n_gold: int

    @property
    def unmatched(self) -> int:
        """Rows left over on either side. Always ``|n_extracted - n_gold|``,
        since matching is greedy over ``min(n_extracted, n_gold)`` pairs."""
        return (self.n_extracted - len(self.pairs)) + (self.n_gold - len(self.pairs))


def _array_group(path: str) -> str | None:
    """The array a field belongs to, or None if its rows cannot be aligned.

    Fields under two levels of array (``a[].b[]``) flatten to lists of lists,
    where a "row" is not a single object, so they stay on positional scoring
    rather than being half-aligned.
    """
    if path.count("[]") != 1:
        return None
    return path[: path.index("[]") + 2]


def _align_arrays(
    schema: ExtractionSchema,
    extracted_flat: dict[str, Any],
    gold_flat: dict[str, Any],
    format_credit: float,
) -> dict[str, _RowAlignment]:
    groups: dict[str, list[str]] = {}
    for spec in schema.fields:
        group = _array_group(spec.path) if spec.in_array else None
        if group is not None:
            groups.setdefault(group, []).append(spec.path)
    return {group: _align_rows(paths, extracted_flat, gold_flat, format_credit) for group, paths in groups.items()}


def _align_rows(
    paths: list[str],
    extracted_flat: dict[str, Any],
    gold_flat: dict[str, Any],
    format_credit: float,
) -> _RowAlignment:
    """Pair extracted rows with gold rows by whole-row similarity.

    Greedy: the most similar pair is taken first, then the next, and so on.
    Optimal assignment would need scipy for a difference that does not show up
    on tables of this size, and greedy keeps the tie-break rule -- lowest gold
    row, then lowest extracted row -- simple enough to be reproducible without
    an RNG. Nothing is left unmatched by choice: exactly ``min(n, m)`` pairs are
    formed, so an unmatched row always means a count disagreement, never a
    similarity that fell below some threshold.
    """
    n_extracted = _row_count(paths, extracted_flat)
    n_gold = _row_count(paths, gold_flat)

    ranked = sorted(
        (-_row_similarity(paths, extracted_flat, gold_flat, i, j, format_credit), j, i)
        for i in range(n_extracted)
        for j in range(n_gold)
    )
    pairs: dict[int, int] = {}
    taken: set[int] = set()
    for _, gold_index, extracted_index in ranked:
        if gold_index in pairs or extracted_index in taken:
            continue
        pairs[gold_index] = extracted_index
        taken.add(extracted_index)
    return _RowAlignment(pairs=pairs, n_extracted=n_extracted, n_gold=n_gold)


def _row_count(paths: list[str], flat: dict[str, Any]) -> int:
    return max((len(flat[p]) for p in paths if isinstance(flat.get(p), list)), default=0)


def _row_similarity(
    paths: list[str],
    extracted_flat: dict[str, Any],
    gold_flat: dict[str, Any],
    extracted_index: int,
    gold_index: int,
    format_credit: float,
) -> float:
    total = 0.0
    for path in paths:
        match compare(_cell(extracted_flat.get(path), extracted_index), _cell(gold_flat.get(path), gold_index)):
            case Match.EXACT:
                total += _ALIGN_EXACT
            case Match.NORMALIZED:
                total += _ALIGN_NORMALIZED
            case Match.DIFFERENT:
                pass
    return total / len(paths) if paths else 0.0


def _cell(column: Any, index: int) -> Any:
    if isinstance(column, list) and index < len(column):
        return column[index]
    return None


def _positional_alignment(n_extracted: int, n_gold: int) -> _RowAlignment:
    overlap = min(n_extracted, n_gold)
    return _RowAlignment(pairs={i: i for i in range(overlap)}, n_extracted=n_extracted, n_gold=n_gold)


def _classify_array(
    *,
    path: str,
    extracted: Any,
    gold: Any,
    gold_flat: dict[str, Any],
    alignment: _RowAlignment | None,
    format_credit: float,
) -> FieldOutcome:
    gold_items = gold if isinstance(gold, list) else []
    extracted_items = extracted if isinstance(extracted, list) else []

    if not gold_items and not extracted_items:
        return FieldOutcome(path=path, score=1.0, error=ErrorClass.CORRECT, extracted=extracted, gold=gold)

    if alignment is None:
        alignment = _positional_alignment(len(extracted_items), len(gold_items))

    scores: list[float] = []
    classes: Counter[ErrorClass] = Counter()
    details: list[str] = []
    related: list[str] = []
    for gold_index in sorted(alignment.pairs):
        score, error, detail, item_related = _classify_scalar(
            path=path,
            extracted=_cell(extracted_items, alignment.pairs[gold_index]),
            gold=_cell(gold_items, gold_index),
            # Sibling scope is the matched gold row, so "took the neighbouring
            # column" stays a claim about one row rather than across rows.
            gold_flat=_item_scope(gold_flat, gold_index),
            format_credit=format_credit,
        )
        scores.append(score)
        classes[error] += 1
        if item_related:
            related.append(item_related)
        if error.is_failure:
            details.append(f"item {gold_index}: {detail}")

    # An unmatched row is a row the extractor invented or dropped. That is a
    # statement about the table, not about this column's value, so it counts as
    # one LENGTH_MISMATCH each rather than as a hallucinated or missing cell.
    if alignment.unmatched:
        classes[ErrorClass.LENGTH_MISMATCH] += alignment.unmatched
        details.insert(0, f"extracted {alignment.n_extracted} items, document contains {alignment.n_gold}")

    denominator = max(alignment.n_extracted, alignment.n_gold)
    failures = [c for c in classes if c.is_failure]
    dominant = max(failures, key=lambda c: classes[c]) if failures else ErrorClass.CORRECT
    return FieldOutcome(
        path=path,
        score=sum(scores) / denominator if denominator else 0.0,
        error=dominant,
        extracted=extracted,
        gold=gold,
        detail="; ".join(details[:4]),
        related_path=related[0] if related else None,
    )


def _item_scope(gold_flat: dict[str, Any], index: int) -> dict[str, Any]:
    """Sibling scope for array item ``index``.

    Within a line-items table, the confusable neighbour of ``quantity`` is
    ``unit_price`` *on the same row* -- so array-valued gold fields are narrowed
    to that row before sibling detection runs.
    """
    scoped: dict[str, Any] = {}
    for path, value in gold_flat.items():
        if isinstance(value, list):
            if index < len(value):
                scoped[path] = value[index]
        else:
            scoped[path] = value
    return scoped

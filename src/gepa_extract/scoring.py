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
) -> DocumentOutcome:
    """Score one extraction against its gold record, field by field."""
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
    outcomes: dict[str, FieldOutcome] = {}
    for spec in schema.fields:
        outcomes[spec.path] = _classify(
            path=spec.path,
            extracted=extracted_flat.get(spec.path),
            gold=gold_flat.get(spec.path),
            gold_flat=gold_flat,
            in_array=spec.in_array,
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
    format_credit: float,
) -> FieldOutcome:
    if in_array:
        return _classify_array(
            path=path, extracted=extracted, gold=gold, gold_flat=gold_flat, format_credit=format_credit
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


def _classify_array(
    *,
    path: str,
    extracted: Any,
    gold: Any,
    gold_flat: dict[str, Any],
    format_credit: float,
) -> FieldOutcome:
    gold_items = gold if isinstance(gold, list) else []
    extracted_items = extracted if isinstance(extracted, list) else []

    if not gold_items and not extracted_items:
        return FieldOutcome(path=path, score=1.0, error=ErrorClass.CORRECT, extracted=extracted, gold=gold)

    if len(gold_items) != len(extracted_items):
        # Length disagreement makes positional scoring meaningless, so score the
        # overlap and report the count mismatch as the headline problem.
        overlap = min(len(gold_items), len(extracted_items))
        matched = sum(
            1 for i in range(overlap) if compare(extracted_items[i], gold_items[i]) is not Match.DIFFERENT
        )
        denominator = max(len(gold_items), len(extracted_items))
        return FieldOutcome(
            path=path,
            score=matched / denominator if denominator else 0.0,
            error=ErrorClass.LENGTH_MISMATCH,
            extracted=extracted,
            gold=gold,
            detail=f"extracted {len(extracted_items)} items, document contains {len(gold_items)}",
        )

    scores: list[float] = []
    classes: Counter[ErrorClass] = Counter()
    details: list[str] = []
    related: list[str] = []
    for index, (item_extracted, item_gold) in enumerate(zip(extracted_items, gold_items, strict=True)):
        score, error, detail, item_related = _classify_scalar(
            path=path,
            extracted=item_extracted,
            gold=item_gold,
            gold_flat=_item_scope(gold_flat, index),
            format_credit=format_credit,
        )
        scores.append(score)
        classes[error] += 1
        if item_related:
            related.append(item_related)
        if error.is_failure:
            details.append(f"item {index}: {detail}")

    failures = [c for c in classes if c.is_failure]
    dominant = max(failures, key=lambda c: classes[c]) if failures else ErrorClass.CORRECT
    return FieldOutcome(
        path=path,
        score=sum(scores) / len(scores),
        error=dominant,
        extracted=extracted,
        gold=gold,
        detail="; ".join(details[:3]),
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

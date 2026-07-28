"""The GEPA seam.

This module and ``optimize.py`` are the only places ``gepa`` is imported.
Everything else -- schema, scoring, extraction, rendering, reflection text --
is a plain domain layer with no knowledge that an optimiser exists. If GEPA's
API moves, or we ever want a different search engine, the blast radius is
these two files and the domain layer ports unchanged.

Two things here are load-bearing:

*Objectives are fields.* ``EvaluationBatch.objective_scores`` carries the
per-field score map, which lets GEPA maintain a Pareto front over
(document x field) rather than over a single mean. A candidate that is the
best in the population at ``total`` survives even if its average is mediocre.

*Reflection sees the document.* Records carry the rendered pages as
``gepa.Image``, so the strong reflection model localises the value visually and
writes that localisation down for the weaker extraction model.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from gepa import Image
from gepa.core.adapter import EvaluationBatch

from gepa_extract.documents import Example
from gepa_extract.errors import ErrorClass
from gepa_extract.extraction import ExtractionResult, Extractor
from gepa_extract.rendering import NullRenderer, PageRenderer
from gepa_extract.schema import ExtractionSchema
from gepa_extract.scoring import DocumentOutcome, score_document

__all__ = ["ExtractionAdapter", "ExtractionTrace"]


@dataclass(slots=True)
class ExtractionTrace:
    """Everything ``make_reflective_dataset`` needs about one rollout."""

    example: Example
    result: ExtractionResult
    outcome: DocumentOutcome


class ExtractionAdapter:
    """Evaluates candidate field descriptions against gold extractions.

    Args:
        schema: The frozen skeleton plus seed descriptions.
        extractor: Backend that runs the extraction.
        renderer: Rasterises documents for the reflection model. Defaults to
            ``NullRenderer`` (text-only reflection), so a caller who has not
            opted into rendering never silently pays for images.
        max_pages: Page cap per document in reflection. Without coordinates we
            cannot know which page holds a field, so all pages are sent by
            default; cap this on long documents.
        max_records_per_component: Cap on reflective records per field. Failures
            are chosen before successes, and ties break on doc_id so runs are
            reproducible without an RNG.
        max_image_documents: How many of those records carry page images. Images
            dominate reflection cost; the rest contribute text evidence only.
        max_workers: Concurrent extractions.
        array_order: Row-matching policy for repeating tables. See
            ``scoring.score_document``.
    """

    def __init__(
        self,
        schema: ExtractionSchema,
        extractor: Extractor,
        *,
        renderer: PageRenderer | None = None,
        max_pages: int | None = None,
        max_records_per_component: int = 6,
        max_image_documents: int = 3,
        max_workers: int = 8,
        array_order: str = "aligned",
    ) -> None:
        self.schema = schema
        self.extractor = extractor
        self.renderer = renderer or NullRenderer()
        self.max_pages = max_pages
        self.max_records_per_component = max_records_per_component
        self.max_image_documents = max_image_documents
        self.max_workers = max_workers
        self.array_order = array_order
        # GEPA discovers this attribute by duck typing; None means "use the
        # built-in reflective proposer", which is what we want -- we steer it
        # with reflection_prompt_template instead of replacing it.
        self.propose_new_texts = None

    # ------------------------------------------------------------------ eval

    def evaluate(
        self,
        batch: list[Example],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        bound_schema = self.schema.bind(candidate)

        def run(example: Example) -> ExtractionTrace:
            try:
                result = self.extractor.extract(
                    document=example.document,
                    schema=bound_schema,
                    task_prompt=self._task_prompt(),
                )
            except Exception as exc:  # noqa: BLE001 - one bad document must not end the run
                result = ExtractionResult.failure(f"{type(exc).__name__}: {exc}")

            outcome = score_document(
                self.schema,
                doc_id=example.doc_id,
                extracted=result.data,
                gold=example.gold,
                extraction_error=result.error,
                array_order=self.array_order,
            )
            return ExtractionTrace(example=example, result=result, outcome=outcome)

        if self.max_workers > 1 and len(batch) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                traces = list(pool.map(run, batch))
        else:
            traces = [run(example) for example in batch]

        return EvaluationBatch(
            outputs=[self._output(trace) for trace in traces],
            scores=[trace.outcome.score for trace in traces],
            trajectories=traces if capture_traces else None,
            objective_scores=[trace.outcome.objective_scores for trace in traces],
        )

    def _task_prompt(self) -> str:
        if self.schema.policy_text:
            return f"{self.schema.task_prompt}\n\n{self.schema.policy_text}"
        return self.schema.task_prompt

    @staticmethod
    def _output(trace: ExtractionTrace) -> dict[str, Any]:
        """Compact, JSON-clean summary. GEPA writes outputs to the run directory,
        so this must not carry document bytes or non-serialisable objects."""
        return {
            "doc_id": trace.example.doc_id,
            "score": round(trace.outcome.score, 4),
            "extracted": trace.result.data,
            "error": trace.result.error,
            "field_scores": {p: round(s, 4) for p, s in trace.outcome.objective_scores.items()},
            "field_errors": {
                p: outcome.error.value for p, outcome in trace.outcome.fields.items() if outcome.is_failure
            },
        }

    # ------------------------------------------------------------ reflection

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        traces: list[ExtractionTrace] = list(eval_batch.trajectories or [])
        dataset: dict[str, list[dict[str, Any]]] = {}

        for component in components_to_update:
            if component not in self.schema.field_paths:
                continue
            dataset[component] = self._records_for(component, traces)
        return dataset

    def _records_for(self, component: str, traces: list[ExtractionTrace]) -> list[dict[str, Any]]:
        failing = [t for t in traces if t.outcome.fields[component].is_failure]
        passing = [t for t in traces if not t.outcome.fields[component].is_failure]

        if not failing:
            # Nothing to learn from here. Sending a pile of successful
            # extractions would spend reflection tokens inviting the model to
            # rewrite a description that is working.
            return [
                {
                    "Document": "(none)",
                    "Feedback": (
                        f"This field was extracted correctly on every document in this batch "
                        f"({len(passing)}/{len(traces)}). The current description is working; change it only if "
                        f"you can see a specific weakness that would fail on a differently laid-out document."
                    ),
                }
            ]

        # Failures first -- they carry the signal. A few successes follow so the
        # model can see what the current description already gets right and
        # avoid regressing it.
        selected = sorted(failing, key=lambda t: t.example.doc_id)[: self.max_records_per_component]
        remaining = self.max_records_per_component - len(selected)
        if remaining > 0:
            selected += sorted(passing, key=lambda t: t.example.doc_id)[:remaining]

        pattern = _pattern_summary(component, traces)
        records: list[dict[str, Any]] = []
        images_left = self.max_image_documents

        for trace in selected:
            outcome = trace.outcome.fields[component]
            attach_images = outcome.is_failure and images_left > 0
            record: dict[str, Any] = {
                "Document": trace.example.doc_id,
                "Extracted value": _render_value(outcome.extracted),
                "Gold value": _render_value(outcome.gold),
                "Result": outcome.error.value,
                "Feedback": _feedback(outcome, pattern),
            }
            if attach_images:
                pages = self._images_for(trace.example)
                if pages:
                    record["Document pages"] = pages
                    images_left -= 1
            records.append(record)

        return records

    def _images_for(self, example: Example) -> list[Image]:
        pages = self.renderer.render(example.document, max_pages=self.max_pages)
        # media_type is passed explicitly: gepa's Image infers image/png for any
        # unrecognised extension without raising, which would silently mislabel
        # whatever we hand it.
        return [Image(path=str(page.path), media_type=page.media_type) for page in pages]


def _feedback(outcome: Any, pattern: str) -> str:
    parts: list[str] = []
    if outcome.detail:
        parts.append(outcome.detail.capitalize() if outcome.detail[:1].islower() else outcome.detail)
    guidance = outcome.error.guidance
    if guidance:
        parts.append(guidance)
    if pattern:
        parts.append(pattern)
    return "\n\n".join(parts) if parts else "Extracted correctly."


def _pattern_summary(component: str, traces: list[ExtractionTrace]) -> str:
    """Aggregate this field's failures across the batch.

    A single wrong document invites an overfitted fix. "6/10 documents: took the
    gold value of 'subtotal'" tells the reflection model it is looking at a
    systematic pattern worth writing a general rule for.
    """
    total = len(traces)
    if not total:
        return ""

    classes: Counter[ErrorClass] = Counter()
    siblings: Counter[str] = Counter()
    for trace in traces:
        outcome = trace.outcome.fields[component]
        if not outcome.is_failure:
            continue
        classes[outcome.error] += 1
        if outcome.error is ErrorClass.SIBLING_VALUE and outcome.related_path:
            siblings[outcome.related_path] += 1

    if not classes:
        return f"Pattern across this batch: correct on all {total} documents."

    lines = [
        f"Pattern across this batch: {count}/{total} documents failed with '{error.value}'."
        for error, count in classes.most_common()
    ]
    lines += [f"Of those, {count} returned the value belonging to '{name}'." for name, count in siblings.most_common()]
    return " ".join(lines)


def _render_value(value: Any) -> Any:
    """Keep list values readable but bounded in the prompt."""
    if isinstance(value, list) and len(value) > 12:
        return value[:12] + [f"... ({len(value) - 12} more)"]
    return value

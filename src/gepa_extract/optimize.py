"""Running an optimisation, and reading the result.

Thin on purpose: the interesting decisions are configuration, and they are
made explicit here rather than left to GEPA's defaults.

``frontier_type="objective"`` is the important one. GEPA's default is
``"instance"``, a Pareto front over documents. Ours is a front over *fields*,
which is what keeps a candidate that is uniquely good at one field alive in
the population even when its mean score is unremarkable. It requires
``objective_scores`` from the adapter, which is exactly what we produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gepa

from gepa_extract.adapter import ExtractionAdapter
from gepa_extract.documents import Example
from gepa_extract.extraction import Extractor
from gepa_extract.reflection import build_templates
from gepa_extract.rendering import PageRenderer
from gepa_extract.schema import ExtractionSchema

__all__ = ["OptimizationResult", "optimize_descriptions", "score_candidate"]


@dataclass(slots=True)
class OptimizationResult:
    seed_descriptions: dict[str, str]
    best_descriptions: dict[str, str]
    seed_score: float
    best_score: float
    gepa_result: Any = None
    per_field_seed: dict[str, float] = field(default_factory=dict)
    per_field_best: dict[str, float] = field(default_factory=dict)

    @property
    def changed_fields(self) -> list[str]:
        return [k for k, v in self.best_descriptions.items() if v.strip() != self.seed_descriptions.get(k, "").strip()]

    def report(self) -> str:
        lines = [
            (
                f"overall   {self.seed_score:.3f} -> {self.best_score:.3f}  "
                f"({self.best_score - self.seed_score:+.3f})"
            ),
            "",
            "per field:",
        ]
        for path in sorted(self.per_field_best):
            before = self.per_field_seed.get(path, 0.0)
            after = self.per_field_best[path]
            marker = "*" if path in self.changed_fields else " "
            lines.append(f"  {marker} {path:<28} {before:.3f} -> {after:.3f}  ({after - before:+.3f})")
        lines += ["", f"descriptions changed: {len(self.changed_fields)}/{len(self.best_descriptions)}"]
        return "\n".join(lines)


def score_candidate(
    schema: ExtractionSchema,
    extractor: Extractor,
    examples: list[Example],
    candidate: dict[str, str],
    *,
    max_workers: int = 8,
) -> tuple[float, dict[str, float]]:
    """Score one candidate over ``examples``. Returns (mean score, per-field means).

    Use this for holdout evaluation -- optimisation reports scores on the sets it
    optimised against, which are not an estimate of generalisation.
    """
    adapter = ExtractionAdapter(schema, extractor, max_workers=max_workers)
    batch = adapter.evaluate(examples, candidate, capture_traces=False)

    per_field: dict[str, list[float]] = {path: [] for path in schema.field_paths}
    for scores in batch.objective_scores or []:
        for path, value in scores.items():
            per_field[path].append(value)

    overall = sum(batch.scores) / len(batch.scores) if batch.scores else 0.0
    means = {path: (sum(v) / len(v) if v else 0.0) for path, v in per_field.items()}
    return overall, means


def optimize_descriptions(
    schema: ExtractionSchema,
    extractor: Extractor,
    trainset: list[Example],
    *,
    reflection_lm: Any,
    valset: list[Example] | None = None,
    renderer: PageRenderer | None = None,
    max_metric_calls: int = 150,
    reflection_minibatch_size: int = 5,
    max_pages: int | None = None,
    max_image_documents: int = 3,
    max_workers: int = 8,
    run_dir: str | None = None,
    seed: int = 0,
    display_progress_bar: bool = False,
    **gepa_kwargs: Any,
) -> OptimizationResult:
    """Evolve the field descriptions of ``schema`` against gold extractions.

    Args:
        reflection_lm: The strong model that writes new descriptions. Must be a
            vision model if ``renderer`` is set, and either a litellm model name
            or a callable taking ``str | list[dict]`` and returning ``str``.
        display_progress_bar: Off by default -- GEPA raises ImportError rather
            than degrading when tqdm is absent, and tqdm is not a dependency of
            this package. Enable it only where tqdm is installed.
        max_metric_calls: Budget, counted in *document extractions*. Every call
            is a real vision request against a real document, so this is the
            knob that decides what a run costs.
    """
    adapter = ExtractionAdapter(
        schema,
        extractor,
        renderer=renderer,
        max_pages=max_pages,
        max_image_documents=max_image_documents,
        max_workers=max_workers,
    )
    seed_candidate = schema.seed_candidate()

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset or trainset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_prompt_template=build_templates(schema),
        # Fields are objectives: keep the front over fields, not over documents.
        frontier_type="objective",
        candidate_selection_strategy="pareto",
        # Round-robin gives every field a turn. With many fields and a small
        # budget, most of the budget goes to evaluation rather than proposal.
        module_selector="round_robin",
        reflection_minibatch_size=reflection_minibatch_size,
        max_metric_calls=max_metric_calls,
        # Repeated (candidate, document) pairs are common once the front has a
        # few members; caching keeps them from being re-extracted.
        cache_evaluation=True,
        run_dir=run_dir,
        seed=seed,
        display_progress_bar=display_progress_bar,
        **gepa_kwargs,
    )

    best = dict(getattr(result, "best_candidate", None) or seed_candidate)
    evaluation_set = valset or trainset
    seed_score, seed_fields = score_candidate(schema, extractor, evaluation_set, seed_candidate, max_workers=max_workers)
    best_score, best_fields = score_candidate(schema, extractor, evaluation_set, best, max_workers=max_workers)

    return OptimizationResult(
        seed_descriptions=seed_candidate,
        best_descriptions=best,
        seed_score=seed_score,
        best_score=best_score,
        gepa_result=result,
        per_field_seed=seed_fields,
        per_field_best=best_fields,
    )

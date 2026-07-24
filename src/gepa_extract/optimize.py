"""Running an optimisation, and reading the result.

Thin on purpose: the interesting decisions are configuration, and they are
made explicit here rather than left to GEPA's defaults.

``frontier_type="objective"`` is the important one. GEPA's default is
``"instance"``, a Pareto front over documents. Ours is a front over *fields*,
which is what keeps a candidate that is uniquely good at one field alive in
the population even when its mean score is unremarkable. It requires
``objective_scores`` from the adapter, which is exactly what we produce.

The second is ``rounds_per_field``. GEPA's native budget, ``max_metric_calls``,
is denominated in document extractions and says nothing about how many fields
get a turn -- on a wide schema most of them never do. ``rounds_per_field`` is
denominated in *rounds per field* instead, so a caller with 100 fields can ask
for two attempts at each and get them. See ``selectors.py`` for the guarantee
and its limits. The two are composable: keep ``max_metric_calls`` as a cost
ceiling and let ``rounds_per_field`` decide coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gepa
from gepa.utils.stop_condition import MaxCandidateProposalsStopper

from gepa_extract.adapter import ExtractionAdapter
from gepa_extract.documents import Example
from gepa_extract.extraction import Extractor
from gepa_extract.reflection import build_templates
from gepa_extract.rendering import PageRenderer
from gepa_extract.schema import ExtractionSchema
from gepa_extract.selectors import FieldCoverageSelector, RoundsPerFieldStopper

__all__ = ["OptimizationResult", "estimate_metric_calls", "optimize_descriptions", "score_candidate"]


@dataclass(slots=True)
class OptimizationResult:
    seed_descriptions: dict[str, str]
    best_descriptions: dict[str, str]
    seed_score: float
    best_score: float
    gepa_result: Any = None
    per_field_seed: dict[str, float] = field(default_factory=dict)
    per_field_best: dict[str, float] = field(default_factory=dict)
    # How many reflection rounds each field actually received. Empty unless
    # rounds_per_field was used -- stock round_robin does not expose this, and
    # inferring it from gepa's trace would be guesswork.
    rounds: dict[str, int] = field(default_factory=dict)

    @property
    def changed_fields(self) -> list[str]:
        return [k for k, v in self.best_descriptions.items() if v.strip() != self.seed_descriptions.get(k, "").strip()]

    def assembled_schema(self, schema: ExtractionSchema) -> dict[str, Any]:
        """The optimised JSON Schema, ready to hand to an extractor.

        ``best_descriptions`` is a flat ``{field_path: description}`` mapping --
        the optimiser's candidate, not something a model can be called with.
        This is the shippable form.

        The schema is a parameter rather than state on this object. The result
        is what gets pickled and written to ``run_dir``, and it has no need to
        carry the skeleton around to do it. Keeping the two separate is also the
        same split the optimiser itself runs on: descriptions travel, structure
        does not.
        """
        return schema.bind(self.best_descriptions)

    @property
    def unvisited_fields(self) -> list[str]:
        """Fields that never got a reflection round. The honest denominator for
        'why did nothing improve here' -- untouched is not the same as tried."""
        if not self.rounds:
            return []
        return [path for path in sorted(self.best_descriptions) if not self.rounds.get(path)]

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
            rounds = f"  [{self.rounds[path]} rounds]" if path in self.rounds else ""
            lines.append(f"  {marker} {path:<28} {before:.3f} -> {after:.3f}  ({after - before:+.3f}){rounds}")
        lines += ["", f"descriptions changed: {len(self.changed_fields)}/{len(self.best_descriptions)}"]
        if self.rounds:
            lines.append(f"fields never given a round: {len(self.unvisited_fields)}/{len(self.best_descriptions)}")
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


def estimate_metric_calls(
    n_unsolved_fields: int,
    n_valset: int,
    *,
    rounds_per_field: int = 1,
    reflection_minibatch_size: int = 5,
    acceptance_rate: float = 0.3,
) -> int:
    """Document-extraction cost of a ``rounds_per_field`` run::

        3*V  +  rounds * unsolved * (2*M + p*V)

    where V is the valset size, M the reflection minibatch size, and p the
    fraction of proposals accepted. The terms:

    * ``3*V`` -- gepa's seed evaluation, plus the two passes
      ``optimize_descriptions`` makes after the run to report seed and best
      scores. Those two are outside gepa's budget entirely and are easy to
      forget; on a short run they dominate everything else.
    * ``2*M`` per round -- gepa evaluates the parent and then the child on the
      round's minibatch.
    * ``p*V`` per round -- an accepted child is re-evaluated on the whole
      valset. This is the only estimated term; the rest is exact.

    Verified against 18 real runs (1-6 unsolved fields x 1-3 rounds x minibatch
    2/4): with ``acceptance_rate=0`` the formula reproduces actual extraction
    counts exactly, so error is confined to how well ``acceptance_rate``
    matches a given corpus. Real runs accept more early and less later, so
    treat the default as a planning ceiling rather than a forecast.

    ``n_unsolved_fields`` is the count of fields that still score below 1.0 --
    **not** the schema's field count. Solved fields retire without consuming
    rounds, so a 100-field schema with 12 bad fields costs what 12 costs. Get
    the number from ``score_candidate`` on the seed candidate before deciding a
    budget.
    """
    per_round = 2 * reflection_minibatch_size + acceptance_rate * n_valset
    return int(3 * n_valset + rounds_per_field * n_unsolved_fields * per_round)


def optimize_descriptions(
    schema: ExtractionSchema,
    extractor: Extractor,
    trainset: list[Example],
    *,
    reflection_lm: Any,
    valset: list[Example] | None = None,
    renderer: PageRenderer | None = None,
    max_metric_calls: int | None = 150,
    rounds_per_field: int | None = None,
    max_iterations: int | None = None,
    reflection_minibatch_size: int = 5,
    max_pages: int | None = None,
    max_image_documents: int = 3,
    max_workers: int = 8,
    module_selector: Any = None,
    frontier_type: str = "objective",
    candidate_selection_strategy: str = "pareto",
    cache_evaluation: bool = True,
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
            knob that decides what a run costs. Pass None to let
            ``rounds_per_field`` alone decide when the run ends -- coverage is
            then guaranteed but cost is not bounded.
        rounds_per_field: Guarantee each field that still needs work this many
            reflection rounds before stopping. This is the knob to reach for on
            a wide schema: ``max_metric_calls`` cannot express "every field gets
            a turn", because it is denominated in documents, not fields. Fields
            that reach a perfect score retire and hand their turns to fields
            still failing, so the cost is set by how many fields are actually
            broken rather than by how many exist.
        max_iterations: Backstop when ``rounds_per_field`` is set. Defaults to
            twice the nominal requirement, which absorbs the iterations gepa
            spends without reaching the selector. Reaching it means the coverage
            guarantee was *not* met; inspect
            ``OptimizationResult.unvisited_fields``.
        module_selector: Overrides field selection entirely. Keeps gepa's
            parameter name because it is a straight passthrough -- gepa's own
            class is ``ReflectionComponentSelector``, and a component is what
            this package calls a field. Leave as None for coverage-guaranteed
            selection (with ``rounds_per_field``) or gepa's ``"round_robin"``
            (without).
    """
    if rounds_per_field is not None and rounds_per_field < 1:
        raise ValueError("rounds_per_field must be at least 1.")
    if max_metric_calls is None and rounds_per_field is None:
        raise ValueError("Provide at least one of max_metric_calls or rounds_per_field as a stopping condition.")
    adapter = ExtractionAdapter(
        schema,
        extractor,
        renderer=renderer,
        max_pages=max_pages,
        max_image_documents=max_image_documents,
        max_workers=max_workers,
    )
    seed_candidate = schema.seed_candidate()

    selector = module_selector
    stop_callbacks: list[Any] = []
    coverage: FieldCoverageSelector | None = None

    if rounds_per_field is not None:
        if selector is None:
            coverage = FieldCoverageSelector()
            selector = coverage
            stop_callbacks.append(RoundsPerFieldStopper(coverage, rounds_per_field))
        # A caller-supplied selector owns its own policy; we cannot promise
        # coverage on its behalf, so rounds_per_field degrades to an iteration
        # cap rather than silently claiming a guarantee it is not enforcing.
        if max_iterations is None:
            max_iterations = 2 * rounds_per_field * len(schema.fields)
        stop_callbacks.append(MaxCandidateProposalsStopper(max_iterations))
    elif max_iterations is not None:
        stop_callbacks.append(MaxCandidateProposalsStopper(max_iterations))

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset or trainset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_prompt_template=build_templates(schema),
        # Fields are objectives: keep the front over fields, not over documents.
        frontier_type=frontier_type,
        candidate_selection_strategy=candidate_selection_strategy,
        # Without rounds_per_field this is gepa's round_robin, whose cursor is
        # per-candidate and so covers fields unevenly once the front grows.
        module_selector=selector if selector is not None else "round_robin",
        reflection_minibatch_size=reflection_minibatch_size,
        max_metric_calls=max_metric_calls,
        stop_callbacks=stop_callbacks or None,
        # Repeated (candidate, document) pairs are common once the front has a
        # few members; caching keeps them from being re-extracted.
        cache_evaluation=cache_evaluation,
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
        rounds={path: coverage.rounds[path] for path in schema.field_paths} if coverage else {},
    )

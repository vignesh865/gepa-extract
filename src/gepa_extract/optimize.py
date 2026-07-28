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

__all__ = [
    "BudgetPlan",
    "OptimizationResult",
    "estimate_metric_calls",
    "optimize_descriptions",
    "plan_budget",
    "score_candidate",
]


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
            (f"overall   {self.seed_score:.3f} -> {self.best_score:.3f}  ({self.best_score - self.seed_score:+.3f})"),
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


@dataclass(slots=True)
class BudgetPlan:
    """Suggested settings for a run, with the reasoning attached."""

    rounds_per_field: int
    valset_size: int
    holdout_size: int
    reflection_minibatch_size: int
    max_metric_calls: int
    estimated_calls: int
    unsolved_assumed: int
    notes: list[str] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        return self.rounds_per_field >= 1

    def explain(self) -> str:
        if not self.fits:
            # No settings are recommended in this case, so none are shown --
            # printing a max_metric_calls here would read as advice to proceed.
            lines = [
                "No affordable plan: the ceiling does not cover even one round.",
                "",
                f"cheapest possible run       ~{self.estimated_calls} document extractions",
                (
                    f"                            (1 round over {self.unsolved_assumed} fields, "
                    f"valset {self.valset_size}, minibatch {self.reflection_minibatch_size})"
                ),
                "",
            ]
        else:
            lines = [
                f"rounds_per_field           {self.rounds_per_field}",
                f"reflection_minibatch_size  {self.reflection_minibatch_size}",
                f"max_metric_calls           {self.max_metric_calls}",
                f"valset / holdout           {self.valset_size} / {self.holdout_size}",
                "",
                f"estimated cost             ~{self.estimated_calls} document extractions",
                f"                           ~{self.rounds_per_field * self.unsolved_assumed} reflection calls",
                "",
            ]
        lines += [f"- {note}" for note in self.notes]
        return "\n".join(lines)


def plan_budget(
    n_fields: int,
    n_documents: int,
    *,
    n_unsolved_fields: int | None = None,
    max_extractions: int | None = None,
    acceptance_rate: float = 0.3,
) -> BudgetPlan:
    """Suggest run settings for a schema of ``n_fields`` and ``n_documents`` docs.

    Advisory, not authoritative: these are defensible starting points, not
    tuned values. The one input that matters most is ``n_unsolved_fields``, and
    it is the one you have to measure rather than guess -- cost scales with the
    fields that are actually broken, and on a typical schema that is a small
    fraction of the total::

        _, per_field = score_candidate(schema, extractor, docs, schema.seed_candidate())
        unsolved = sum(1 for s in per_field.values() if s < 1.0)
        plan = plan_budget(len(schema.fields), len(docs), n_unsolved_fields=unsolved)

    Left unspecified, every field is assumed broken, which is a ceiling rather
    than a forecast -- usually a large overestimate.

    Args:
        max_extractions: A ceiling you are willing to spend. Given one, rounds
            are solved for rather than assumed, and ``fits`` reports False when
            not even a single round is affordable.
    """
    if n_fields < 1 or n_documents < 2:
        raise ValueError("Need at least 1 field and 2 documents to plan a run.")

    notes: list[str] = []
    unsolved = n_unsolved_fields if n_unsolved_fields is not None else n_fields
    if n_unsolved_fields is None:
        notes.append(
            f"Assuming all {n_fields} fields need work, which is a ceiling. Score the seed candidate "
            f"first and pass n_unsolved_fields -- it is usually far smaller, and cost scales with it."
        )

    # Hold out roughly a fifth for honest final scoring, but only when there is
    # enough left to optimise against. Below ~10 documents a holdout costs more
    # in signal than it buys in confidence.
    holdout = max(2, n_documents // 5) if n_documents >= 10 else 0
    valset = n_documents - holdout
    if holdout == 0:
        notes.append(
            f"Too few documents ({n_documents}) to hold any out; scores will be measured on the "
            f"documents optimised against and will overstate generalisation."
        )
    elif valset < 8:
        notes.append(f"A {valset}-document valset is small; one unusual layout can steer the whole run.")

    # Minibatch drives 2M of the per-round cost, so it is the cheap lever. Small
    # batches make the per-field diagnosis noisier, hence the floor.
    minibatch = min(5, max(2, valset // 4))

    def cost(rounds: int) -> int:
        return estimate_metric_calls(
            unsolved,
            valset,
            rounds_per_field=rounds,
            reflection_minibatch_size=minibatch,
            acceptance_rate=acceptance_rate,
        )

    # Two rounds is the recommendation: one gives a field no recovery from a
    # single bad proposal, and past two the returns fall off sharply. A ceiling
    # can only push this down, never up -- it is a limit on what may be spent,
    # not an instruction to spend it, and a third round costs ~50% more for a
    # gain that is usually marginal.
    rounds = 2
    if max_extractions is not None:
        affordable = [r for r in (2, 1) if cost(r) <= max_extractions]
        rounds = affordable[0] if affordable else 0
        if rounds == 0:
            notes.append(
                f"Even one round costs ~{cost(1)}, above the {max_extractions} ceiling. Reduce the field "
                f"count with n_unsolved_fields, shrink the valset, or raise the ceiling."
            )
        elif rounds == 1:
            notes.append("Only one round is affordable: a field that gets a bad proposal cannot recover from it.")

    estimated = cost(max(rounds, 1))
    return BudgetPlan(
        rounds_per_field=rounds,
        valset_size=valset,
        holdout_size=holdout,
        reflection_minibatch_size=minibatch,
        # Headroom over the estimate: acceptance_rate is the one term that is
        # guessed, and a run that accepts more than assumed costs more.
        max_metric_calls=int(estimated * 1.5),
        estimated_calls=estimated,
        unsolved_assumed=unsolved,
        notes=notes,
    )


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
    array_order: str = "aligned",
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
        array_order: How rows of a repeating table are paired with gold rows
            when scoring. Defaults to ``"aligned"``: rows are matched by
            content, so an extractor that reorders or drops a row is charged
            for that row alone instead of having every subsequent row
            misclassified. Use ``"positional"`` only when document order is
            itself part of the contract.
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
        array_order=array_order,
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
    seed_score, seed_fields = score_candidate(
        schema, extractor, evaluation_set, seed_candidate, max_workers=max_workers
    )
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

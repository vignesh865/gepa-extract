"""The coverage guarantee.

The promise ``rounds_per_field`` makes is narrow and worth stating exactly:
every field that has not yet scored perfectly is *selected* for reflection at
least that many times. Not that the proposals are good, and not that they are
accepted -- selection is the only thing the optimiser controls.

These tests pin that promise, and equally pin what it is not: a solved field
stops consuming turns, and a run whose fields are all solved terminates rather
than spinning.
"""

from __future__ import annotations

import pytest

from gepa_extract import (
    FieldCoverageSelector,
    RoundsPerFieldStopper,
    StubExtractor,
    optimize_descriptions,
)
from gepa_extract.optimize import estimate_metric_calls, plan_budget
from tests.conftest import make_gold
from tests.test_gepa_contracts import IMPROVED_TOTAL, FakeReflectionLM, subtotal_grabbing_extractor


class FakeState:
    """The two attributes the selector reads off gepa's state."""

    def __init__(self, objective_scores=None, candidates=None):
        self.prog_candidate_objective_scores = objective_scores or []
        self.program_candidates = candidates or []


CANDIDATE = {"a": "", "b": "", "c": ""}


class TestSelectionIsFairAndDeterministic:
    def test_it_cycles_least_served_first(self) -> None:
        selector = FieldCoverageSelector()
        state = FakeState()
        picks = [selector(state, [], [], 0, CANDIDATE)[0] for _ in range(6)]
        assert picks == ["a", "b", "c", "a", "b", "c"]
        assert dict(selector.rounds) == {"a": 2, "b": 2, "c": 2}

    def test_ties_break_on_path_not_at_random(self) -> None:
        """Reproducibility without threading an RNG through the optimiser."""
        runs = []
        for _ in range(3):
            selector = FieldCoverageSelector()
            state = FakeState()
            runs.append([selector(state, [], [], 0, CANDIDATE)[0] for _ in range(5)])
        assert runs[0] == runs[1] == runs[2]

    def test_coverage_survives_the_parent_candidate_changing(self) -> None:
        """The counter is global. gepa's own round_robin keeps its cursor per
        candidate, so alternating parents makes it revisit the same field; this
        selector must not care which parent the Pareto front handed it."""
        selector = FieldCoverageSelector()
        state = FakeState()
        picks = [selector(state, [], [], idx % 3, CANDIDATE)[0] for idx in range(6)]
        assert sorted(picks) == ["a", "a", "b", "b", "c", "c"]


class TestSolvedFieldsRetire:
    def test_a_perfect_field_stops_consuming_turns(self) -> None:
        selector = FieldCoverageSelector()
        state = FakeState(objective_scores=[{"a": 1.0, "b": 0.4, "c": 0.2}])
        picks = [selector(state, [], [], 0, CANDIDATE)[0] for _ in range(4)]
        assert "a" not in picks, "a is already perfect; its turns belong to b and c"
        assert picks == ["b", "c", "b", "c"]

    def test_retirement_is_permanent(self) -> None:
        """Judging eligibility against the *current* parent would let a field
        re-enter after a regression elsewhere on the front, and the stopper's
        min() could then never rise -- a run that cannot terminate."""
        selector = FieldCoverageSelector()
        perfect = FakeState(objective_scores=[{"a": 1.0}])
        assert "a" not in selector.eligible(perfect, CANDIDATE)

        regressed = FakeState(objective_scores=[{"a": 0.1}])
        assert "a" not in selector.eligible(regressed, CANDIDATE), "retirement must not be revocable"

    def test_a_field_perfect_on_any_candidate_counts_as_solved(self) -> None:
        selector = FieldCoverageSelector()
        state = FakeState(objective_scores=[{"a": 0.2}, {"a": 1.0}])
        assert selector.eligible(state, CANDIDATE) == ["b", "c"]

    def test_all_fields_solved_still_returns_a_selection(self) -> None:
        """Returning [] would have gepa build an empty reflective dataset and
        burn the iteration. The stopper normally ends the run first."""
        selector = FieldCoverageSelector()
        state = FakeState(objective_scores=[{"a": 1.0, "b": 1.0, "c": 1.0}])
        assert selector(state, [], [], 0, CANDIDATE) == ["a"]


class TestStopper:
    def test_it_waits_for_every_unsolved_field(self) -> None:
        selector = FieldCoverageSelector()
        stopper = RoundsPerFieldStopper(selector, rounds_per_field=2)
        state = FakeState(candidates=[CANDIDATE])

        for _ in range(6):  # 3 fields x 2 rounds
            assert stopper(state) is False, "some field is still short of its 2 rounds"
            selector(state, [], [], 0, CANDIDATE)
        assert stopper(state) is True

    def test_solved_fields_are_not_waited_on(self) -> None:
        selector = FieldCoverageSelector()
        stopper = RoundsPerFieldStopper(selector, rounds_per_field=1)
        state = FakeState(objective_scores=[{"c": 1.0}], candidates=[CANDIDATE])

        selector(state, [], [], 0, CANDIDATE)
        assert stopper(state) is False
        selector(state, [], [], 0, CANDIDATE)
        assert stopper(state) is True, "c is solved; a and b have each had their round"

    def test_it_stops_when_everything_is_solved(self) -> None:
        selector = FieldCoverageSelector()
        stopper = RoundsPerFieldStopper(selector, rounds_per_field=99)
        state = FakeState(objective_scores=[{"a": 1.0, "b": 1.0, "c": 1.0}], candidates=[CANDIDATE])
        assert stopper(state) is True

    def test_it_does_not_stop_before_the_seed_is_evaluated(self) -> None:
        assert RoundsPerFieldStopper(FieldCoverageSelector(), 1)(FakeState()) is False

    def test_zero_rounds_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            RoundsPerFieldStopper(FieldCoverageSelector(), 0)


def unfixable_extractor(examples) -> StubExtractor:
    """Breaks three fields in three different ways, with no description that
    repairs them -- so nothing ever retires and the full guarantee is testable."""
    return StubExtractor(
        responses={
            e.doc_id: {
                **make_gold(e.doc_id),
                "total": 1200.00,  # sibling_value: the subtotal
                "invoice_date": "03/14/2024",  # format_mismatch
                "purchase_order": "PO-9001",  # hallucinated (gold is None)
            }
            for e in examples
        }
    )


BROKEN = ("total", "invoice_date", "purchase_order")


class TestEndToEndCoverage:
    """The claim a caller actually relies on, through a real gepa run."""

    def test_every_unsolved_field_gets_its_rounds(self, schema, examples) -> None:
        result = optimize_descriptions(
            schema,
            unfixable_extractor(examples),
            trainset=examples,
            reflection_lm=FakeReflectionLM("A description that fixes nothing."),
            rounds_per_field=2,
            max_metric_calls=None,
            reflection_minibatch_size=2,
            seed=0,
        )

        for path in BROKEN:
            assert result.per_field_best[path] < 1.0, "precondition: this field must stay broken"
            assert result.rounds[path] >= 2, f"{path} was promised 2 rounds, got {result.rounds[path]}"

    def test_fields_that_are_already_perfect_never_consume_a_round(self, schema, examples) -> None:
        result = optimize_descriptions(
            schema,
            unfixable_extractor(examples),
            trainset=examples,
            reflection_lm=FakeReflectionLM("A description that fixes nothing."),
            rounds_per_field=2,
            max_metric_calls=None,
            reflection_minibatch_size=2,
            seed=0,
        )

        for path in schema.field_paths:
            if path not in BROKEN:
                assert result.rounds[path] == 0, f"{path} scored 1.0 on the seed and should never be picked"
        assert set(result.unvisited_fields) == set(schema.field_paths) - set(BROKEN)

    def test_a_field_solved_early_stops_consuming_rounds(self, schema, examples) -> None:
        """The other half of "if it needs to be optimized": ``total`` is fixed by
        its first proposal, so it retires having used one round of the two it was
        nominally owed. Spending the second on a solved field is the waste this
        is meant to avoid."""
        result = optimize_descriptions(
            schema,
            subtotal_grabbing_extractor(examples),
            trainset=examples,
            reflection_lm=FakeReflectionLM(IMPROVED_TOTAL),
            rounds_per_field=2,
            max_metric_calls=None,
            reflection_minibatch_size=2,
            seed=0,
        )
        assert result.per_field_best["total"] == 1.0
        assert result.rounds["total"] == 1

    def test_without_rounds_per_field_coverage_is_not_claimed(self, schema, examples) -> None:
        """The old behaviour still works, and honestly reports no round data
        rather than fabricating it from gepa's trace."""
        result = optimize_descriptions(
            schema,
            subtotal_grabbing_extractor(examples),
            trainset=examples,
            reflection_lm=FakeReflectionLM(IMPROVED_TOTAL),
            max_metric_calls=40,
            reflection_minibatch_size=2,
            seed=0,
        )
        assert result.rounds == {}
        assert result.unvisited_fields == []

    def test_a_narrow_budget_leaves_most_fields_untouched(self, schema, examples) -> None:
        """The problem this exists to solve, demonstrated: with a metric-call
        budget that buys only a couple of iterations, most fields never get a
        turn -- which is invisible unless someone counts."""
        result = optimize_descriptions(
            schema,
            subtotal_grabbing_extractor(examples),
            trainset=examples,
            reflection_lm=FakeReflectionLM(IMPROVED_TOTAL),
            rounds_per_field=1,
            max_metric_calls=18,
            reflection_minibatch_size=2,
            seed=0,
        )
        assert result.unvisited_fields, "a starved run must report the fields it never reached"

    def test_rejects_a_run_with_no_stopping_condition(self, schema, examples) -> None:
        with pytest.raises(ValueError, match="stopping condition"):
            optimize_descriptions(
                schema,
                subtotal_grabbing_extractor(examples),
                trainset=examples,
                reflection_lm=FakeReflectionLM(IMPROVED_TOTAL),
                max_metric_calls=None,
            )


class TestCostEstimate:
    def test_coverage_and_cost_are_different_currencies(self) -> None:
        """The trap the estimator exists to close: '2 rounds over 100 fields'
        sounds like 200, and is not."""
        assert estimate_metric_calls(100, 10, rounds_per_field=2, reflection_minibatch_size=3) > 1500

    def test_it_scales_with_the_fields_that_need_work(self) -> None:
        wide = estimate_metric_calls(100, 10, rounds_per_field=2)
        narrow = estimate_metric_calls(15, 10, rounds_per_field=2)
        assert narrow < wide / 5

    def test_it_counts_the_post_run_scoring_passes(self) -> None:
        """optimize_descriptions scores seed and best after the run, outside
        gepa's budget. Omitting them understated a short run by more than half:
        the 1-field/1-round/minibatch-2 case below is 40 extractions, 24 of
        which are these passes."""
        assert estimate_metric_calls(0, 12, rounds_per_field=1) == 36

    def test_planning_advice_scales_with_the_broken_fields_not_the_schema(self) -> None:
        """The point of measuring first: the same schema and corpus, planned
        with and without knowing how many fields are actually broken."""
        blind = plan_budget(120, 40)
        measured = plan_budget(120, 40, n_unsolved_fields=18)
        assert measured.estimated_calls < blind.estimated_calls / 5
        assert any("ceiling" in note for note in blind.notes), "an assumed field count must be flagged"
        assert measured.notes == [], "a measured plan within budget needs no caveats"

    def test_a_ceiling_trades_rounds_down_before_giving_up(self) -> None:
        assert plan_budget(120, 40, n_unsolved_fields=18, max_extractions=500).rounds_per_field == 1
        assert plan_budget(120, 40, n_unsolved_fields=18, max_extractions=900).rounds_per_field == 2

    def test_a_generous_ceiling_is_a_limit_not_a_target(self) -> None:
        """A ceiling may push rounds down, never up. Spending a large budget on
        a third round costs ~50% more for a gain that is usually marginal."""
        assert plan_budget(120, 40, n_unsolved_fields=18, max_extractions=100_000).rounds_per_field == 2
        assert plan_budget(120, 40, n_unsolved_fields=18).rounds_per_field == 2

    def test_an_unaffordable_ceiling_reports_rather_than_recommends(self) -> None:
        """Silently returning rounds=1 over budget would be worse than useless:
        the caller asked for a ceiling and would blow through it."""
        plan = plan_budget(120, 40, max_extractions=500)
        assert plan.fits is False
        assert plan.rounds_per_field == 0
        explained = plan.explain()
        assert "No affordable plan" in explained
        assert "max_metric_calls" not in explained, "must not read as advice to proceed"

    def test_a_corpus_too_small_to_split_says_so(self) -> None:
        plan = plan_budget(10, 6)
        assert plan.holdout_size == 0
        assert plan.valset_size == 6
        assert any("overstate generalisation" in note for note in plan.notes)
        assert plan.reflection_minibatch_size <= plan.valset_size

    def test_it_refuses_inputs_it_cannot_plan_for(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            plan_budget(0, 20)
        with pytest.raises(ValueError, match="at least"):
            plan_budget(10, 1)

    def test_the_recommended_ceiling_leaves_headroom_over_the_estimate(self) -> None:
        """acceptance_rate is the one guessed term; a run that accepts more than
        assumed costs more, and the ceiling must not cut it off at exactly the
        estimate."""
        plan = plan_budget(120, 40, n_unsolved_fields=18)
        assert plan.max_metric_calls > plan.estimated_calls

    def test_the_formula_matches_measured_runs_when_nothing_is_accepted(self) -> None:
        """Calibration, pinned. Against 18 real runs over the 12-document
        corpus, with a reflection LM that never improves anything (so nothing is
        accepted and the p*V term is genuinely zero), the estimate equalled the
        actual extraction count exactly. These four are spot checks from that
        sweep; if the arithmetic drifts, the documented budgets are wrong."""
        measured = [
            # (unsolved, rounds, minibatch, actual extractions)
            (1, 1, 2, 40),
            (3, 2, 4, 84),
            (6, 2, 2, 84),
            (6, 3, 4, 180),
        ]
        for unsolved, rounds, minibatch, actual in measured:
            predicted = estimate_metric_calls(
                unsolved,
                12,
                rounds_per_field=rounds,
                reflection_minibatch_size=minibatch,
                acceptance_rate=0.0,
            )
            assert predicted == actual, f"{unsolved} fields, {rounds} rounds, mb={minibatch}"

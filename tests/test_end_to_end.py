"""The whole loop, offline.

This is the test that would catch a pipeline that is wired up but inert: one
where descriptions evolve, scores are computed, and nothing actually improves
because the two are not connected.

`StubExtractor.description_rules` is what makes it meaningful -- the stub
returns the subtotal for `total` until the description mentions "beneath the
tax line", at which point it returns the correct value. That models the real
relationship (a better description produces a better extraction) without a
network call, so the run can only succeed if feedback genuinely reaches the
reflection model and its output genuinely reaches the extractor.
"""

from __future__ import annotations

from gepa_extract import ExtractionAdapter, StubExtractor, optimize_descriptions, score_candidate
from tests.conftest import make_gold
from tests.test_gepa_contracts import IMPROVED_TOTAL, FakeReflectionLM


def build_extractor(examples) -> StubExtractor:
    return StubExtractor(
        responses={e.doc_id: {**make_gold(e.doc_id), "total": 1200.00} for e in examples},
        description_rules={"total": [("beneath the tax line", {"total": 1320.00})]},
    )


def test_optimisation_repairs_the_field_it_was_given_evidence_about(schema, examples) -> None:
    extractor = build_extractor(examples)
    result = optimize_descriptions(
        schema,
        extractor,
        trainset=examples,
        reflection_lm=FakeReflectionLM(IMPROVED_TOTAL),
        max_metric_calls=120,
        reflection_minibatch_size=3,
        display_progress_bar=False,
        seed=0,
    )

    assert result.per_field_seed["total"] == 0.0
    assert result.per_field_best["total"] == 1.0
    assert result.best_score > result.seed_score
    assert "total" in result.changed_fields


def test_fields_that_were_already_correct_are_not_regressed(schema, examples) -> None:
    result = optimize_descriptions(
        schema,
        build_extractor(examples),
        trainset=examples,
        reflection_lm=FakeReflectionLM(IMPROVED_TOTAL),
        max_metric_calls=120,
        reflection_minibatch_size=3,
        display_progress_bar=False,
        seed=0,
    )
    for path, before in result.per_field_seed.items():
        assert result.per_field_best[path] >= before, f"{path} regressed"


def test_report_is_readable(schema, examples) -> None:
    result = optimize_descriptions(
        schema,
        build_extractor(examples),
        trainset=examples,
        reflection_lm=FakeReflectionLM(IMPROVED_TOTAL),
        max_metric_calls=60,
        reflection_minibatch_size=3,
        display_progress_bar=False,
        seed=0,
    )
    report = result.report()
    assert "overall" in report
    assert "total" in report


def test_the_assembled_schema_carries_the_evolved_descriptions(schema, examples) -> None:
    """The shippable artifact: descriptions dropped back into the skeleton, with
    structure provably untouched. This is what an extractor is called with --
    best_descriptions on its own is the optimiser's candidate, not a schema."""
    result = optimize_descriptions(
        schema,
        build_extractor(examples),
        trainset=examples,
        reflection_lm=FakeReflectionLM(IMPROVED_TOTAL),
        max_metric_calls=120,
        reflection_minibatch_size=3,
        display_progress_bar=False,
        seed=0,
    )

    assembled = result.assembled_schema(schema)
    assert assembled["properties"]["total"]["description"] == result.best_descriptions["total"]
    assert "beneath the tax line" in assembled["properties"]["total"]["description"]
    # Every leaf carries a description, including the fields never proposed for:
    # a partial candidate must not leave holes in the schema handed to a model.
    assert assembled["properties"]["subtotal"]["description"] == result.best_descriptions["subtotal"]
    assert schema.structural_fingerprint_of(assembled) == schema.fingerprint()


def test_the_reflection_model_is_shown_the_sibling_diagnosis(schema, examples) -> None:
    """The point of per-field feedback: the prompt must say which field's value
    was taken, not merely that the answer was wrong."""
    reflection_lm = FakeReflectionLM(IMPROVED_TOTAL)
    optimize_descriptions(
        schema,
        build_extractor(examples),
        trainset=examples,
        reflection_lm=reflection_lm,
        max_metric_calls=120,
        reflection_minibatch_size=3,
        display_progress_bar=False,
        seed=0,
    )

    prompts = [p if isinstance(p, str) else str(p) for p in reflection_lm.prompts]
    total_prompts = [p for p in prompts if "'subtotal'" in p]
    assert total_prompts, "no reflection prompt carried the sibling diagnosis"
    assert any("documents failed with 'sibling_value'" in p for p in total_prompts)
    assert any("SMALLER, CHEAPER model" in p for p in prompts), "asymmetry framing missing from the prompt"


def test_holdout_scoring_uses_unseen_documents(schema, examples) -> None:
    """Optimisation reports scores on the set it optimised against; generalisation
    has to be measured separately."""
    train, holdout = examples[:3], examples[3:]
    extractor = build_extractor(examples)

    result = optimize_descriptions(
        schema,
        extractor,
        trainset=train,
        reflection_lm=FakeReflectionLM(IMPROVED_TOTAL),
        max_metric_calls=100,
        reflection_minibatch_size=3,
        display_progress_bar=False,
        seed=0,
    )

    before, _ = score_candidate(schema, extractor, holdout, result.seed_descriptions)
    after, per_field = score_candidate(schema, extractor, holdout, result.best_descriptions)
    assert after > before
    assert per_field["total"] == 1.0


def test_adapter_evaluate_is_reusable_after_optimisation(schema, examples) -> None:
    extractor = build_extractor(examples)
    result = optimize_descriptions(
        schema,
        extractor,
        trainset=examples,
        reflection_lm=FakeReflectionLM(IMPROVED_TOTAL),
        max_metric_calls=60,
        reflection_minibatch_size=3,
        display_progress_bar=False,
        seed=0,
    )
    batch = ExtractionAdapter(schema, extractor).evaluate(examples, result.best_descriptions)
    assert len(batch.scores) == len(examples)

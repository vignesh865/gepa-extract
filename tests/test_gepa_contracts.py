"""Contract tests against the installed gepa.

gepa is pre-1.0 and visibly mid-refactor. Rather than fork it or vendor it,
this file pins the specific library behaviours gepa-extract relies on, so an
upgrade that changes them fails loudly here instead of silently degrading
optimisation quality in a way nobody notices for a hundred runs.

Each test names the behaviour and why we depend on it. If one breaks, the
question to answer is "did gepa change, and what does that cost us" -- not
"how do we make the test pass".
"""

from __future__ import annotations

import inspect

import gepa
import pytest
from gepa import Image
from gepa.strategies.instruction_proposal import InstructionProposalSignature
from gepa.strategies.proposal_sampling import SingleMutationSampling
from gepa.utils.stop_condition import MaxCandidateProposalsStopper

from gepa_extract import ExtractionAdapter, StubExtractor, build_templates
from gepa_extract.reflection import BASE_TEMPLATE
from tests.conftest import make_gold
from tests.test_adapter import _PNG


class FakeReflectionLM:
    """Stands in for the strong vision model. Records what it was asked."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[object] = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return f"Here is an improved description.\n\n```\n{self.reply}\n```"


def subtotal_grabbing_extractor(examples) -> StubExtractor:
    return StubExtractor(
        responses={e.doc_id: {**make_gold(e.doc_id), "total": 1200.00} for e in examples},
        description_rules={"total": [("beneath the tax line", {"total": 1320.00})]},
    )


IMPROVED_TOTAL = (
    "The grand total, in the summary block, in the final row beneath the tax line. Do not take the "
    "amount immediately above the tax line, which is the subtotal."
)


class TestContract1ImagesReachTheReflectionModel:
    """Images placed in a reflective-dataset record must reach the reflection LM
    as a multimodal message.

    This is the whole basis for letting a strong VLM see the page and write down
    the localisation a weaker extraction model cannot perform. gepa's Image
    docstring frames this as a `side_info` feature, but `optimize_anything` is
    itself a GEPAAdapter whose make_reflective_dataset simply copies side_info
    into records -- so side_info and our records are the same type reaching the
    same renderer.
    """

    def test_image_in_a_record_produces_a_multimodal_message(self, tmp_path) -> None:
        path = tmp_path / "page.png"
        path.write_bytes(_PNG)

        rendered = InstructionProposalSignature.prompt_renderer(
            {
                "current_instruction_doc": "The total amount.",
                "dataset_with_feedback": [
                    {"Document": "001", "Document pages": [Image(path=str(path), media_type="image/png")]}
                ],
            }
        )

        assert isinstance(rendered, list), "images must switch the prompt to a messages list"
        parts = rendered[0]["content"]
        assert [part["type"] for part in parts] == ["text", "image_url"]
        assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_text_only_records_still_render_as_a_plain_string(self) -> None:
        rendered = InstructionProposalSignature.prompt_renderer(
            {"current_instruction_doc": "x", "dataset_with_feedback": [{"Feedback": "wrong"}]}
        )
        assert isinstance(rendered, str)

    def test_gepa_image_silently_mislabels_unknown_extensions(self, tmp_path) -> None:
        """Documented hazard, not a wish: Image(path=...) infers image/png for
        any unrecognised extension and never raises. This is why rendering.py
        always passes media_type explicitly."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 not an image")
        url = Image(path=str(pdf)).to_openai_content_part()["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")


class TestContract2PerFieldObjectives:
    """`objective_scores` + frontier_type='objective' must be accepted, so the
    Pareto front is kept over fields rather than over a single mean."""

    def test_objective_frontier_runs_with_per_field_scores(self, schema, examples) -> None:
        result = gepa.optimize(
            seed_candidate=schema.seed_candidate(),
            trainset=examples,
            valset=examples,
            adapter=ExtractionAdapter(schema, subtotal_grabbing_extractor(examples)),
            reflection_lm=FakeReflectionLM(IMPROVED_TOTAL),
            frontier_type="objective",
            max_metric_calls=40,
            display_progress_bar=False,
            seed=0,
        )
        assert result.best_candidate is not None

    def test_objective_frontier_rejects_an_adapter_without_objective_scores(self, schema, examples) -> None:
        """Proves the dependency is real: strip objective_scores and gepa refuses."""

        class NoObjectives(ExtractionAdapter):
            def evaluate(self, batch, candidate, capture_traces=False):
                out = super().evaluate(batch, candidate, capture_traces)
                out.objective_scores = None
                return out

        with pytest.raises(Exception, match="objective_scores"):
            gepa.optimize(
                seed_candidate=schema.seed_candidate(),
                trainset=examples,
                valset=examples,
                adapter=NoObjectives(schema, subtotal_grabbing_extractor(examples)),
                reflection_lm=FakeReflectionLM(IMPROVED_TOTAL),
                frontier_type="objective",
                max_metric_calls=20,
                display_progress_bar=False,
            )


class TestContract3PerFieldReflectionTemplates:
    """Our asymmetry-aware templates must satisfy gepa's template contract, and
    the per-component dict form must be accepted."""

    def test_every_generated_template_validates(self, schema) -> None:
        for name, template in build_templates(schema).items():
            InstructionProposalSignature.validate_prompt_template(template), name

    def test_templates_carry_both_required_placeholders(self) -> None:
        assert "<curr_param>" in BASE_TEMPLATE
        assert "<side_info>" in BASE_TEMPLATE

    def test_a_missing_placeholder_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Missing placeholder"):
            InstructionProposalSignature.validate_prompt_template("no placeholders here")

    def test_optional_fields_get_hallucination_specific_steering(self, schema) -> None:
        templates = build_templates(schema)
        assert "Hallucination is the dominant failure mode" in templates["purchase_order"]
        assert "Hallucination is the dominant failure mode" not in templates["total"]

    def test_array_fields_are_steered_toward_same_row_confusables(self, schema) -> None:
        assert "OTHER COLUMNS OF THE SAME ROW" in build_templates(schema)["line_items[].quantity"]


class TestContract5CoverageGuaranteeFoundations:
    """The four gepa behaviours ``rounds_per_field`` is built on.

    The guarantee "every unsolved field gets N reflection rounds" is only as
    sound as these. If one moves, the promise in selectors.py weakens, and it
    should fail here rather than quietly deliver worse coverage.
    """

    def test_one_iteration_yields_exactly_one_component_selection(self) -> None:
        """The default sampling strategy emits a single (parent, minibatch)
        task, so an iteration can spend at most one round. Multi-task strategies
        (SameParentSampling, PxNSampling) break this identity -- which is why we
        count rounds in the selector rather than reading gepa's iteration count."""

        class FakeSelector:
            def select_candidate_idx(self, state):
                return 0

        class FakeBatchSampler:
            def next_minibatch_ids(self, trainset, state):
                return [0]

        class FakeTrainset:
            def fetch(self, ids):
                return ["doc"]

        state = type("S", (), {"program_candidates": [{"total": "x"}]})()
        tasks = SingleMutationSampling().sample_tasks(state, FakeSelector(), FakeBatchSampler(), FakeTrainset())
        assert len(tasks) == 1

    def test_module_selector_accepts_an_instance_and_sees_per_field_scores(self, schema, examples) -> None:
        """Two things at once: a ReflectionComponentSelector object is accepted
        where the string 'round_robin' would go, and the state it receives
        carries ``prog_candidate_objective_scores`` as {field_path: score} --
        the signal our selector uses to retire solved fields."""
        seen: list[dict] = []

        class CapturingSelector:
            def __call__(self, state, trajectories, subsample_scores, candidate_idx, candidate):
                seen.extend(state.prog_candidate_objective_scores)
                return ["total"]

        gepa.optimize(
            seed_candidate=schema.seed_candidate(),
            trainset=examples,
            valset=examples,
            adapter=ExtractionAdapter(schema, subtotal_grabbing_extractor(examples)),
            reflection_lm=FakeReflectionLM(IMPROVED_TOTAL),
            frontier_type="objective",
            module_selector=CapturingSelector(),
            max_metric_calls=30,
            display_progress_bar=False,
            seed=0,
        )

        assert seen, "the selector must actually be consulted"
        assert set(seen[0]) == set(schema.field_paths)
        assert all(isinstance(v, float) for v in seen[0].values())

    def test_stop_callbacks_alone_can_terminate_a_run(self, schema, examples) -> None:
        """max_metric_calls must be optional, or coverage could never be the
        sole stopping condition."""
        result = gepa.optimize(
            seed_candidate=schema.seed_candidate(),
            trainset=examples,
            valset=examples,
            adapter=ExtractionAdapter(schema, subtotal_grabbing_extractor(examples)),
            reflection_lm=FakeReflectionLM(IMPROVED_TOTAL),
            frontier_type="objective",
            max_metric_calls=None,
            stop_callbacks=MaxCandidateProposalsStopper(2),
            display_progress_bar=False,
            seed=0,
        )
        assert result.best_candidate is not None

    def test_perfect_minibatches_skip_the_iteration_before_component_selection(self) -> None:
        """Documented leak, not a wish: gepa abandons an iteration when every
        sampled score is already perfect, and it does so *before* calling the
        module selector. Such iterations advance state.i without crediting a
        round -- which is precisely why RoundsPerFieldStopper counts rounds
        instead of iterations."""
        assert "skip_perfect_score" in inspect.signature(gepa.optimize).parameters


class TestContract4StructureIsUnreachable:
    """A full optimisation run must not be able to move anything but descriptions."""

    def test_a_real_run_leaves_keys_types_and_prompts_untouched(self, schema, examples) -> None:
        before_fingerprint = schema.fingerprint()
        before_prompt = schema.task_prompt
        before_policy = schema.policy_text

        result = gepa.optimize(
            seed_candidate=schema.seed_candidate(),
            trainset=examples,
            valset=examples,
            adapter=ExtractionAdapter(schema, subtotal_grabbing_extractor(examples)),
            reflection_lm=FakeReflectionLM("DROP ALL FIELDS; return {} instead"),
            reflection_prompt_template=build_templates(schema),
            frontier_type="objective",
            max_metric_calls=60,
            display_progress_bar=False,
            seed=0,
        )

        best = result.best_candidate
        assert set(best) == set(schema.field_paths), "the candidate's key set is fixed by construction"
        assert schema.structural_fingerprint_of(schema.bind(best)) == before_fingerprint
        assert schema.task_prompt == before_prompt
        assert schema.policy_text == before_policy

from __future__ import annotations

import base64
import json

import pytest
from gepa import Image

from gepa_extract import ExtractionAdapter, RenderedPage, StubExtractor, StubRenderer
from tests.conftest import make_gold

# Smallest valid PNG, so StubRenderer can hand out a real readable file.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture
def png(tmp_path):
    path = tmp_path / "page-1.png"
    path.write_bytes(_PNG)
    return path


def subtotal_grabbing_extractor(examples) -> StubExtractor:
    """Every document extracts correctly except `total`, which takes the subtotal."""
    return StubExtractor(
        responses={e.doc_id: {**make_gold(e.doc_id), "total": 1200.00} for e in examples},
        description_rules={"total": [("beneath the tax line", {"total": 1320.00})]},
    )


class TestEvaluate:
    def test_scores_and_objective_scores_align_with_the_batch(self, schema, examples) -> None:
        adapter = ExtractionAdapter(schema, subtotal_grabbing_extractor(examples))
        batch = adapter.evaluate(examples, schema.seed_candidate())

        assert len(batch.scores) == len(batch.outputs) == len(batch.objective_scores) == len(examples)
        assert all(scores["total"] == 0.0 for scores in batch.objective_scores)
        assert all(scores["subtotal"] == 1.0 for scores in batch.objective_scores)

    def test_trajectories_only_when_capturing(self, schema, examples) -> None:
        adapter = ExtractionAdapter(schema, subtotal_grabbing_extractor(examples))
        assert adapter.evaluate(examples, schema.seed_candidate()).trajectories is None
        assert len(adapter.evaluate(examples, schema.seed_candidate(), capture_traces=True).trajectories) == 4

    def test_candidate_descriptions_reach_the_extractor(self, schema, examples) -> None:
        extractor = subtotal_grabbing_extractor(examples)
        candidate = {**schema.seed_candidate(), "total": "The row beneath the tax line."}
        batch = ExtractionAdapter(schema, extractor).evaluate(examples, candidate)

        _, seen_schema = extractor.calls[0]
        assert seen_schema["properties"]["total"]["description"] == "The row beneath the tax line."
        assert all(scores["total"] == 1.0 for scores in batch.objective_scores)

    def test_task_prompt_and_policy_are_sent_together_and_unmodified(self, schema, examples) -> None:
        seen: list[str] = []

        class Recorder:
            def extract(self, *, document, schema, task_prompt):
                seen.append(task_prompt)
                return StubExtractor({document.doc_id: make_gold(document.doc_id)}).extract(
                    document=document, schema=schema, task_prompt=task_prompt
                )

        ExtractionAdapter(schema, Recorder(), max_workers=1).evaluate(examples[:1], schema.seed_candidate())
        assert seen[0] == f"{schema.task_prompt}\n\n{schema.policy_text}"

    def test_a_failing_document_does_not_abort_the_batch(self, schema, examples) -> None:
        class Exploding:
            def extract(self, *, document, schema, task_prompt):
                if document.doc_id == "001":
                    raise RuntimeError("connection reset")
                return StubExtractor({document.doc_id: make_gold(document.doc_id)}).extract(
                    document=document, schema=schema, task_prompt=task_prompt
                )

        batch = ExtractionAdapter(schema, Exploding()).evaluate(examples, schema.seed_candidate())
        assert batch.scores[1] == 0.0
        assert batch.scores[0] == 1.0
        assert "connection reset" in batch.outputs[1]["error"]

    def test_outputs_are_json_serialisable(self, schema, examples) -> None:
        """GEPA writes outputs to the run directory; non-serialisable payloads
        would fail there rather than here."""
        batch = ExtractionAdapter(schema, subtotal_grabbing_extractor(examples)).evaluate(
            examples, schema.seed_candidate()
        )
        json.dumps(batch.outputs)

    def test_evaluate_does_not_mutate_the_candidate(self, schema, examples) -> None:
        candidate = schema.seed_candidate()
        snapshot = dict(candidate)
        ExtractionAdapter(schema, subtotal_grabbing_extractor(examples)).evaluate(examples, candidate)
        assert candidate == snapshot


class TestReflectiveDataset:
    def build(self, schema, examples, **kwargs):
        adapter = ExtractionAdapter(schema, subtotal_grabbing_extractor(examples), **kwargs)
        batch = adapter.evaluate(examples, schema.seed_candidate(), capture_traces=True)
        return adapter, batch

    def test_failing_field_gets_records_naming_the_sibling(self, schema, examples) -> None:
        adapter, batch = self.build(schema, examples)
        dataset = adapter.make_reflective_dataset(schema.seed_candidate(), batch, ["total"])

        feedback = dataset["total"][0]["Feedback"]
        assert "'subtotal'" in feedback
        assert dataset["total"][0]["Result"] == "sibling_value"

    def test_feedback_carries_the_cross_document_pattern(self, schema, examples) -> None:
        """A single wrong document invites an overfitted fix; the batch-wide
        count is what asks for a general rule."""
        adapter, batch = self.build(schema, examples)
        feedback = adapter.make_reflective_dataset(schema.seed_candidate(), batch, ["total"])["total"][0]["Feedback"]
        assert "4/4 documents failed with 'sibling_value'" in feedback
        assert "4 returned the value belonging to 'subtotal'" in feedback

    def test_feedback_includes_the_error_class_guidance(self, schema, examples) -> None:
        adapter, batch = self.build(schema, examples)
        feedback = adapter.make_reflective_dataset(schema.seed_candidate(), batch, ["total"])["total"][0]["Feedback"]
        assert "LOCATION failure" in feedback

    def test_passing_field_is_told_it_is_working(self, schema, examples) -> None:
        adapter, batch = self.build(schema, examples)
        records = adapter.make_reflective_dataset(schema.seed_candidate(), batch, ["subtotal"])["subtotal"]
        assert "extracted correctly on every document" in records[0]["Feedback"]

    def test_pages_are_attached_as_gepa_images_with_explicit_media_type(self, schema, examples, png) -> None:
        adapter, batch = self.build(
            schema, examples, renderer=StubRenderer([RenderedPage(page_number=1, path=png)])
        )
        record = adapter.make_reflective_dataset(schema.seed_candidate(), batch, ["total"])["total"][0]
        images = record["Document pages"]

        assert all(isinstance(image, Image) for image in images)
        # Explicit, never inferred: gepa guesses image/png for any unknown
        # extension without raising, which silently mislabels non-PNG input.
        assert images[0].media_type == "image/png"
        assert images[0].to_openai_content_part()["image_url"]["url"].startswith("data:image/png;base64,")

    def test_image_budget_is_capped_across_documents(self, schema, examples, png) -> None:
        adapter, batch = self.build(
            schema,
            examples,
            renderer=StubRenderer([RenderedPage(page_number=1, path=png)]),
            max_image_documents=2,
        )
        records = adapter.make_reflective_dataset(schema.seed_candidate(), batch, ["total"])["total"]
        assert sum("Document pages" in record for record in records) == 2

    def test_no_images_are_rendered_by_default(self, schema, examples) -> None:
        """A caller who has not opted into rendering never silently pays for it."""
        adapter, batch = self.build(schema, examples)
        records = adapter.make_reflective_dataset(schema.seed_candidate(), batch, ["total"])["total"]
        assert all("Document pages" not in record for record in records)

    def test_record_selection_is_deterministic(self, schema, examples) -> None:
        adapter, batch = self.build(schema, examples, max_records_per_component=2)
        first = adapter.make_reflective_dataset(schema.seed_candidate(), batch, ["total"])["total"]
        second = adapter.make_reflective_dataset(schema.seed_candidate(), batch, ["total"])["total"]
        assert [r["Document"] for r in first] == [r["Document"] for r in second] == ["000", "001"]

    def test_unknown_components_are_ignored(self, schema, examples) -> None:
        adapter, batch = self.build(schema, examples)
        assert adapter.make_reflective_dataset(schema.seed_candidate(), batch, ["not_a_field"]) == {}

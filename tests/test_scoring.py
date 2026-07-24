from __future__ import annotations

import pytest

from gepa_extract import ErrorClass, ExtractionSchema, score_document
from gepa_extract.scoring import FORMAT_CREDIT
from gepa_extract.values import Match, compare
from tests.conftest import make_gold


def score(schema: ExtractionSchema, overrides: dict, gold: dict | None = None):
    gold = gold or make_gold("001")
    extracted = {**make_gold("001"), **overrides}
    return score_document(schema, doc_id="001", extracted=extracted, gold=gold)


class TestValueComparison:
    @pytest.mark.parametrize(
        ("extracted", "gold", "expected"),
        [
            (1320.0, 1320.0, Match.EXACT),
            ("$1,320.00", 1320.00, Match.NORMALIZED),
            ("1320", 1320.0, Match.NORMALIZED),
            ("(1,320.00)", -1320.0, Match.NORMALIZED),
            ("1.320,50", 1320.50, Match.NORMALIZED),  # European separators
            (1200.0, 1320.0, Match.DIFFERENT),
            ("03/14/2024", "2024-03-14", Match.NORMALIZED),
            ("March 14, 2024", "2024-03-14", Match.NORMALIZED),
            ("2024-03-15", "2024-03-14", Match.DIFFERENT),
            ("  Acme Industrial Supply ", "Acme Industrial Supply", Match.NORMALIZED),
            ("ACME industrial supply", "Acme Industrial Supply", Match.NORMALIZED),
            ("Acme Industrial", "Acme Industrial Supply", Match.DIFFERENT),
            (None, None, Match.EXACT),
            ("N/A", None, Match.NORMALIZED),
            ("PO-9", None, Match.DIFFERENT),
        ],
    )
    def test_compare(self, extracted, gold, expected) -> None:
        assert compare(extracted, gold) is expected


class TestErrorClasses:
    def test_correct_field_scores_one(self, schema: ExtractionSchema) -> None:
        outcome = score(schema, {})
        assert outcome.score == 1.0
        assert outcome.fields["total"].error is ErrorClass.CORRECT

    def test_total_taking_the_subtotal_is_a_sibling_value(self, schema: ExtractionSchema) -> None:
        """The motivating failure: the extracted value IS another field's gold value."""
        outcome = score(schema, {"total": 1200.00})
        total = outcome.fields["total"]
        assert total.error is ErrorClass.SIBLING_VALUE
        assert total.score == 0.0
        assert "'subtotal'" in total.detail

    def test_wrong_date_format_gets_partial_credit(self, schema: ExtractionSchema) -> None:
        outcome = score(schema, {"invoice_date": "03/14/2024"})
        date_field = outcome.fields["invoice_date"]
        assert date_field.error is ErrorClass.FORMAT_MISMATCH
        assert date_field.score == FORMAT_CREDIT

    def test_absent_optional_field_invented_is_hallucination(self, schema: ExtractionSchema) -> None:
        outcome = score(schema, {"purchase_order": "PO-44192"})
        po = outcome.fields["purchase_order"]
        assert po.error is ErrorClass.HALLUCINATED
        assert po.score == 0.0

    def test_absent_optional_field_left_null_is_correct(self, schema: ExtractionSchema) -> None:
        assert score(schema, {"purchase_order": None}).fields["purchase_order"].error is ErrorClass.CORRECT
        assert score(schema, {"purchase_order": "N/A"}).fields["purchase_order"].error is ErrorClass.CORRECT

    def test_present_field_returned_null_is_missing(self, schema: ExtractionSchema) -> None:
        outcome = score(schema, {"total": None})
        assert outcome.fields["total"].error is ErrorClass.MISSING

    def test_unrelated_value_is_wrong_value(self, schema: ExtractionSchema) -> None:
        outcome = score(schema, {"total": 9999.99})
        assert outcome.fields["total"].error is ErrorClass.WRONG_VALUE

    def test_nested_object_field_is_scored(self, schema: ExtractionSchema) -> None:
        outcome = score(schema, {"vendor": {"name": "Globex"}})
        assert outcome.fields["vendor.name"].error is ErrorClass.WRONG_VALUE


class TestArrayFields:
    def test_missing_row_is_a_length_mismatch(self, schema: ExtractionSchema) -> None:
        outcome = score(schema, {"line_items": [{"description": "Steel bracket", "quantity": 4}]})
        quantity = outcome.fields["line_items[].quantity"]
        assert quantity.error is ErrorClass.LENGTH_MISMATCH
        assert quantity.score == pytest.approx(0.5)
        assert "extracted 1 items, document contains 2" in quantity.detail

    def test_per_item_scoring_is_partial(self, schema: ExtractionSchema) -> None:
        outcome = score(
            schema,
            {
                "line_items": [
                    {"description": "Steel bracket", "quantity": 4},
                    {"description": "Hex bolt, M8", "quantity": 99},
                ]
            },
        )
        assert outcome.fields["line_items[].quantity"].score == pytest.approx(0.5)
        assert outcome.fields["line_items[].description"].score == 1.0

    def test_sibling_detection_is_scoped_to_the_same_row(self, schema: ExtractionSchema) -> None:
        """Taking the neighbouring column of the SAME row must be detected as a
        sibling, not written off as an unrelated wrong value."""
        gold = make_gold("001")
        gold["line_items"] = [
            {"description": "Steel bracket", "quantity": 4},
            {"description": "Hex bolt, M8", "quantity": 40},
        ]
        extracted = {**gold, "line_items": [{"description": "4", "quantity": 4}, gold["line_items"][1]]}
        outcome = score_document(schema, doc_id="001", extracted=extracted, gold=gold)
        assert outcome.fields["line_items[].description"].error is ErrorClass.SIBLING_VALUE
        assert "'line_items[].quantity'" in outcome.fields["line_items[].description"].detail


class TestDocumentOutcome:
    def test_objective_scores_cover_every_field(self, schema: ExtractionSchema) -> None:
        outcome = score(schema, {"total": 1200.00})
        assert set(outcome.objective_scores) == set(schema.field_paths)
        assert outcome.objective_scores["total"] == 0.0
        assert outcome.objective_scores["subtotal"] == 1.0

    def test_document_score_is_the_mean_of_fields(self, schema: ExtractionSchema) -> None:
        outcome = score(schema, {"total": 1200.00})
        assert outcome.score == pytest.approx(8 / 9)

    def test_extraction_failure_zeroes_every_field_and_keeps_the_message(self, schema: ExtractionSchema) -> None:
        outcome = score_document(
            schema, doc_id="001", extracted=None, gold=make_gold("001"), extraction_error="429 rate limited"
        )
        assert outcome.score == 0.0
        assert outcome.extraction_error == "429 rate limited"
        assert "429 rate limited" in outcome.fields["total"].detail

    def test_failures_lists_only_failing_fields(self, schema: ExtractionSchema) -> None:
        outcome = score(schema, {"total": 1200.00, "purchase_order": "PO-1"})
        assert {f.path for f in outcome.failures()} == {"total", "purchase_order"}

"""Backend tests that need no SDK and no API key.

The conversion layers are where backends actually go wrong -- a schema the
provider rejects, or visual evidence silently dropped from a reflection
prompt -- so those are tested directly rather than behind a network call.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from gepa_extract.backends.gemini import GeminiReflectionLM, to_gemini_schema
from tests.conftest import INVOICE_SCHEMA


class FakePart:
    def __init__(self, data: bytes, mime_type: str) -> None:
        self.data = data
        self.mime_type = mime_type


FAKE_TYPES = SimpleNamespace(Part=SimpleNamespace(from_bytes=lambda *, data, mime_type: FakePart(data, mime_type)))


class TestSchemaConversion:
    def test_types_are_uppercased_for_the_openapi_subset(self) -> None:
        converted = to_gemini_schema(INVOICE_SCHEMA)
        assert converted["type"] == "OBJECT"
        assert converted["properties"]["total"]["type"] == "NUMBER"
        assert converted["properties"]["line_items"]["type"] == "ARRAY"

    def test_descriptions_survive_conversion(self) -> None:
        """The descriptions are the payload -- losing them would silently
        optimise text that never reaches the model."""
        converted = to_gemini_schema(INVOICE_SCHEMA)
        assert converted["properties"]["total"]["description"] == "The total amount."
        assert (
            converted["properties"]["line_items"]["items"]["properties"]["quantity"]["description"]
            == "The quantity."
        )

    def test_optional_fields_become_nullable(self) -> None:
        """Without a legal way to say 'absent', a model invents a value."""
        converted = to_gemini_schema(INVOICE_SCHEMA)
        assert converted["properties"]["purchase_order"]["nullable"] is True
        assert "nullable" not in converted["properties"]["total"]

    def test_required_lists_are_preserved(self) -> None:
        converted = to_gemini_schema(INVOICE_SCHEMA)
        assert "total" in converted["required"]
        assert converted["properties"]["line_items"]["items"]["required"] == ["description", "quantity"]

    def test_type_unions_with_null_become_nullable(self) -> None:
        converted = to_gemini_schema({"type": ["string", "null"], "description": "maybe"})
        assert converted == {"type": "STRING", "description": "maybe", "nullable": True}

    def test_unsupported_keys_are_dropped(self) -> None:
        """Gemini rejects unknown schema keys outright."""
        converted = to_gemini_schema(
            {"type": "object", "additionalProperties": False, "$schema": "...", "properties": {}}
        )
        assert set(converted) <= {"type", "description", "enum", "format", "properties", "required", "nullable"}

    def test_nested_objects_recurse(self) -> None:
        converted = to_gemini_schema(INVOICE_SCHEMA)
        assert converted["properties"]["vendor"]["properties"]["name"]["type"] == "STRING"


class TestReflectionPromptConversion:
    def test_a_plain_string_passes_through(self) -> None:
        assert GeminiReflectionLM._to_parts("improve this", FAKE_TYPES) == ["improve this"]

    def test_text_and_image_parts_are_both_converted(self) -> None:
        payload = base64.b64encode(b"\x89PNG fake").decode()
        parts = GeminiReflectionLM._to_parts(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "why did total fail?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}},
                    ],
                }
            ],
            FAKE_TYPES,
        )

        assert parts[0] == "why did total fail?"
        assert isinstance(parts[1], FakePart)
        assert parts[1].mime_type == "image/png"
        assert parts[1].data == b"\x89PNG fake"

    def test_a_non_data_uri_raises_rather_than_dropping_the_evidence(self) -> None:
        """Silently discarding the page image would leave the reflection model
        blind while still appearing to work."""
        with pytest.raises(ValueError, match="base64 data URIs"):
            GeminiReflectionLM._to_parts(
                [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example/p.png"}}]}],
                FAKE_TYPES,
            )

    def test_string_content_messages_are_supported(self) -> None:
        assert GeminiReflectionLM._to_parts([{"role": "user", "content": "hello"}], FAKE_TYPES) == ["hello"]

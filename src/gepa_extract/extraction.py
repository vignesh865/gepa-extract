"""The extraction backend interface.

One method, deliberately. Everything a backend needs -- the bound schema, the
document, the frozen prompt -- is passed in; nothing about GEPA, scoring or
optimisation leaks across this boundary. Adding a provider means implementing
``extract``.

Backends must not raise for a failed document. A malformed response from one
invoice is data about the current candidate, not a reason to abort a run that
may be hours in, so failures come back as ``ExtractionResult.failure(...)``
and are scored as zeros with the error text preserved for reflection.

``extract`` is called concurrently -- see ``ExtractionAdapter(max_workers=...)``
-- so backends must be thread-safe, or the caller must set ``max_workers=1``.
Network-bound backends generally are. Backends that parse PDFs in-process
generally are not: PDFium in particular segfaults the interpreter under
concurrent use rather than raising.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from gepa_extract.documents import Document

__all__ = ["CallableExtractor", "ExtractionResult", "Extractor", "StubExtractor"]


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    data: dict[str, Any] | None
    error: str | None = None
    raw_response: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None and self.data is not None

    @classmethod
    def success(cls, data: dict[str, Any], raw_response: str = "") -> ExtractionResult:
        return cls(data=data, error=None, raw_response=raw_response)

    @classmethod
    def failure(cls, error: str, raw_response: str = "") -> ExtractionResult:
        return cls(data=None, error=error, raw_response=raw_response)


class Extractor(Protocol):
    """Extracts structured data from one document against one bound schema."""

    def extract(
        self,
        *,
        document: Document,
        schema: dict[str, Any],
        task_prompt: str,
    ) -> ExtractionResult: ...


class CallableExtractor:
    """Adapts a plain function into an Extractor."""

    def __init__(self, fn: Callable[[Document, dict[str, Any], str], dict[str, Any] | ExtractionResult]) -> None:
        self._fn = fn

    def extract(self, *, document: Document, schema: dict[str, Any], task_prompt: str) -> ExtractionResult:
        try:
            result = self._fn(document, schema, task_prompt)
        except Exception as exc:  # noqa: BLE001 - a bad document must not end the run
            return ExtractionResult.failure(f"{type(exc).__name__}: {exc}")
        if isinstance(result, ExtractionResult):
            return result
        return ExtractionResult.success(result)


class StubExtractor:
    """A deterministic, API-free extractor for tests and offline examples.

    It is driven by a per-document response table, and optionally by
    ``description_rules``: substrings that, when present in a field's evolved
    description, change what that field returns. That second part is what makes
    an end-to-end optimisation test meaningful -- the loop can only improve if
    better descriptions actually produce better extractions, and this models
    that relationship without a network call.

    Example::

        StubExtractor(
            responses={"inv-001": {"total": "1200.00"}},
            description_rules={"total": [("beneath the tax line", {"total": "1320.00"})]},
        )
    """

    def __init__(
        self,
        responses: dict[str, dict[str, Any]],
        description_rules: dict[str, list[tuple[str, dict[str, Any]]]] | None = None,
    ) -> None:
        self.responses = responses
        self.description_rules = description_rules or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def extract(self, *, document: Document, schema: dict[str, Any], task_prompt: str) -> ExtractionResult:
        self.calls.append((document.doc_id, schema))
        if document.doc_id not in self.responses:
            return ExtractionResult.failure(f"no stub response registered for {document.doc_id!r}")

        data = _deep_copy(self.responses[document.doc_id])
        descriptions = _descriptions_by_path(schema)
        for path, rules in self.description_rules.items():
            description = descriptions.get(path, "")
            for trigger, override in rules:
                if trigger.casefold() in description.casefold():
                    data = _merge(data, _deep_copy(override))
        return ExtractionResult.success(data, raw_response="<stub>")


def _descriptions_by_path(schema: dict[str, Any], prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    node_type = schema.get("type")
    if node_type == "object":
        for key, child in schema.get("properties", {}).items():
            out |= _descriptions_by_path(child, f"{prefix}.{key}" if prefix else key)
    elif node_type == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            out |= _descriptions_by_path(items, f"{prefix}[]")
    elif prefix:
        out[prefix] = schema.get("description", "")
    return out


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value


def _merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = _merge(base.get(key), value) if key in base else value
        return merged
    return override

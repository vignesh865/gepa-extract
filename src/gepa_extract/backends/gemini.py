"""Gemini backends: a weak extractor and a strong vision reflector.

The two halves are deliberately asymmetric, and that asymmetry is the point of
the whole project. The extractor is a cheap model run once per document per
metric call. The reflector is an expensive vision model run once per proposal,
whose job is to look at pages the extractor got wrong and write down the
localisation the extractor could not perform itself.

Requires ``pip install 'gepa-extract[gemini]'`` and ``GOOGLE_API_KEY``.

Note on PDFs: the *extractor* sends the native PDF -- Gemini accepts
``application/pdf`` directly, keeping the text layer. Only the reflection path
rasterises, because GEPA can carry images into a reflection prompt but not
documents.
"""

from __future__ import annotations

import base64
import json
import os
import re
import threading
from typing import Any

from gepa_extract.documents import Document
from gepa_extract.extraction import ExtractionResult

__all__ = ["GeminiExtractor", "GeminiReflectionLM", "to_gemini_schema"]

_DATA_URI = re.compile(r"^data:([^;,]+);base64,(.*)$", re.DOTALL)


def _import_genai():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Gemini backends require the google-genai SDK. Install with: pip install 'gepa-extract[gemini]'"
        ) from exc
    return genai, types


def to_gemini_schema(schema: dict[str, Any], *, required: bool = True) -> dict[str, Any]:
    """Convert a JSON Schema to Gemini's OpenAPI subset.

    Descriptions are preserved verbatim -- they are the payload. Optional fields
    become ``nullable`` so the model has a legal way to say "absent" instead of
    inventing a value. Keys outside the subset (``$schema``,
    ``additionalProperties``, ``examples``) are dropped rather than passed
    through, because the API rejects unknown keys outright.
    """
    node_type = schema.get("type", "string")
    if isinstance(node_type, list):  # ["string", "null"] style unions
        non_null = [t for t in node_type if t != "null"]
        node_type = non_null[0] if non_null else "string"
        required = False

    out: dict[str, Any] = {"type": str(node_type).upper()}
    for key in ("description", "enum", "format"):
        if key in schema:
            out[key] = schema[key]
    if not required:
        out["nullable"] = True

    if node_type == "object":
        required_keys = list(schema.get("required", []))
        properties = schema.get("properties", {})
        out["properties"] = {
            key: to_gemini_schema(child, required=key in required_keys) for key, child in properties.items()
        }
        if required_keys:
            out["required"] = required_keys
    elif node_type == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            out["items"] = to_gemini_schema(items, required=True)
    return out


class GeminiExtractor:
    """Extracts structured data from a document with a Gemini model.

    Args:
        model: Defaults to a cheap model on purpose. If you optimise
            descriptions against a strong extractor, you learn nothing about
            whether the descriptions carry the information a weak one needs.
        temperature: 0 by default. Extraction should be reproducible; sampling
            noise would be scored as if it were candidate quality.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        client: Any = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._client = client
        self._client_lock = threading.Lock()
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")

    @property
    def client(self) -> Any:
        # Double-checked locking. `extract` is called from the adapter's thread
        # pool, and an unguarded lazy init lets several threads each build a
        # Client; the losers are garbage collected, and closing them tears down
        # transport the winner is still using -- surfacing as "Cannot send a
        # request, as the client has been closed" on unrelated documents.
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    genai, _ = _import_genai()
                    self._client = genai.Client(api_key=self._api_key)
        return self._client

    def extract(self, *, document: Document, schema: dict[str, Any], task_prompt: str) -> ExtractionResult:
        _, types = _import_genai()
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=document.read_bytes(), mime_type=document.media_type),
                    task_prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    response_mime_type="application/json",
                    response_schema=to_gemini_schema(schema),
                ),
            )
        except Exception as exc:  # noqa: BLE001 - a failed document is data, not a crash
            return ExtractionResult.failure(f"{type(exc).__name__}: {exc}")

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            return ExtractionResult.failure("model returned an empty response")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            return ExtractionResult.failure(f"response was not valid JSON: {exc}", raw_response=text)
        if not isinstance(data, dict):
            return ExtractionResult.failure(f"expected a JSON object, got {type(data).__name__}", raw_response=text)
        return ExtractionResult.success(data, raw_response=text)


class GeminiReflectionLM:
    """A vision-capable reflection LM conforming to GEPA's ``LanguageModel``.

    GEPA calls this with either a plain string or an OpenAI-style messages list
    whose content parts are ``text`` and ``image_url`` (the latter carrying a
    base64 data URI, since our renderer produces local PNG files). Both shapes
    are translated to Gemini parts here.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        *,
        api_key: str | None = None,
        temperature: float = 1.0,
        client: Any = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._client = client
        self._client_lock = threading.Lock()
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")

    @property
    def client(self) -> Any:
        # Double-checked locking. `extract` is called from the adapter's thread
        # pool, and an unguarded lazy init lets several threads each build a
        # Client; the losers are garbage collected, and closing them tears down
        # transport the winner is still using -- surfacing as "Cannot send a
        # request, as the client has been closed" on unrelated documents.
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    genai, _ = _import_genai()
                    self._client = genai.Client(api_key=self._api_key)
        return self._client

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        _, types = _import_genai()
        parts = self._to_parts(prompt, types)
        response = self.client.models.generate_content(
            model=self.model,
            contents=parts,
            config=types.GenerateContentConfig(temperature=self.temperature),
        )
        return (getattr(response, "text", None) or "").strip()

    @staticmethod
    def _to_parts(prompt: str | list[dict[str, Any]], types: Any) -> list[Any]:
        if isinstance(prompt, str):
            return [prompt]

        parts: list[Any] = []
        for message in prompt:
            content = message.get("content", "")
            if isinstance(content, str):
                parts.append(content)
                continue
            for item in content:
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    match = _DATA_URI.match(url)
                    if match is None:
                        # Remote URLs would need fetching; our renderer never
                        # produces them, so surface it rather than silently drop
                        # the visual evidence the reflection depends on.
                        raise ValueError(f"GeminiReflectionLM expects base64 data URIs, got: {url[:60]}...")
                    media_type, payload = match.group(1), match.group(2)
                    parts.append(types.Part.from_bytes(data=base64.b64decode(payload), mime_type=media_type))
        return parts

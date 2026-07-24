"""Adding an extraction backend.

Run:  python -m examples.generate_invoices
      python -m examples.03_custom_backend

A backend implements one method::

    def extract(self, *, document, schema, task_prompt) -> ExtractionResult

Nothing about GEPA, scoring, or optimisation crosses that boundary, so a
backend is testable on its own and swappable without touching anything else.

The backend built here is a genuinely useful one: instead of sending the PDF to
a vision model, it pulls the PDF's **text layer** and sends text to any chat
LLM. For born-digital documents like these that is a fraction of the cost of a
vision call, and the descriptions being optimised are the same either way.

It runs offline against a scripted LLM so the example works with no API key;
the final section shows the one-line swap to a real provider.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from examples.invoice_schema import build_schema
from gepa_extract import ExtractionResult, load_examples, score_candidate

DATA = Path(__file__).parent / "data" / "invoices.json"


class ChatLLM(Protocol):
    """Any text-in/text-out model."""

    def __call__(self, prompt: str) -> str: ...


class TextLayerExtractor:
    """Extracts from a PDF's text layer via any chat LLM.

    Args:
        llm: Callable taking a prompt and returning the model's text.
        max_characters: Truncation guard. Long documents otherwise blow the
            context window silently, and a truncated prompt produces a
            confidently wrong extraction rather than an error.

    Thread safety: PDFium is not thread-safe, so this backend must be run with
    ``max_workers=1``. Anything higher segfaults the interpreter rather than
    raising. A network-bound backend has no such constraint -- concurrency is a
    per-backend property, which is why it is a parameter and not a default
    baked into the adapter.
    """

    def __init__(self, llm: ChatLLM, *, max_characters: int = 20_000) -> None:
        self.llm = llm
        self.max_characters = max_characters

    def extract(self, *, document, schema: dict[str, Any], task_prompt: str) -> ExtractionResult:
        try:
            text = self._read_text(document)
        except Exception as exc:  # noqa: BLE001 - a bad document is data, not a crash
            return ExtractionResult.failure(f"could not read text layer: {type(exc).__name__}: {exc}")

        if not text.strip():
            # Scanned PDFs have no text layer. Say so precisely: this backend is
            # the wrong tool for that document, which is different from the model
            # failing to find the fields.
            return ExtractionResult.failure("no text layer in document (scanned PDF -- use a vision backend)")

        prompt = (
            f"{task_prompt}\n\n"
            f"Return JSON matching this schema. Each field's 'description' is your instruction for that "
            f"field -- follow it exactly.\n\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            f"--- DOCUMENT TEXT ---\n{text[: self.max_characters]}\n--- END ---\n\n"
            f"Respond with JSON only."
        )

        try:
            raw = self.llm(prompt)
        except Exception as exc:  # noqa: BLE001
            return ExtractionResult.failure(f"{type(exc).__name__}: {exc}")

        return self._parse(raw)

    @staticmethod
    def _read_text(document) -> str:
        import pypdfium2

        pdf = pypdfium2.PdfDocument(document.read_bytes())
        try:
            return "\n".join(page.get_textpage().get_text_range() for page in pdf)
        finally:
            pdf.close()

    @staticmethod
    def _parse(raw: str) -> ExtractionResult:
        text = raw.strip()
        if text.startswith("```"):  # models fence JSON even when told not to
            text = text.split("```", 2)[1]
            text = text.split("\n", 1)[1] if text.lower().startswith("json") else text
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            return ExtractionResult.failure(f"response was not valid JSON: {exc}", raw_response=raw)
        if not isinstance(data, dict):
            return ExtractionResult.failure(f"expected a JSON object, got {type(data).__name__}", raw_response=raw)
        return ExtractionResult.success(data, raw_response=raw)


def scripted_llm(examples) -> Callable[[str], str]:
    """A stand-in LLM that reads the document text, so the example runs offline.

    It finds the invoice number in the prompt, looks up that document's gold
    record, and returns it with `total` corrupted to the subtotal -- the failure
    a small model actually makes on these layouts.
    """
    by_invoice_number = {e.gold["invoice_number"]: e.gold for e in examples}

    def call(prompt: str) -> str:
        for invoice_number, gold in by_invoice_number.items():
            if invoice_number in prompt:
                return json.dumps({**gold, "total": gold["subtotal"]})
        return "not a document I recognise"  # exercises the JSON-failure path

    return call


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Run `python -m examples.generate_invoices` first.")

    schema = build_schema()
    examples = load_examples(DATA)
    extractor = TextLayerExtractor(scripted_llm(examples))

    # max_workers=1: this backend reads PDFs with PDFium, which is not thread-safe.
    overall, per_field = score_candidate(schema, extractor, examples, schema.seed_candidate(), max_workers=1)
    print(f"text-layer backend, seed descriptions: {overall:.3f} overall")
    for path in sorted(per_field):
        print(f"    {path:<28} {per_field[path]:.3f}")

    print("\nA malformed response is scored, not raised:")
    from gepa_extract import Document

    bad = TextLayerExtractor(lambda prompt: "sorry, I can't help with that")
    result = bad.extract(
        document=Document(doc_id="acme-00", path=DATA.parent / "acme-00.pdf"),
        schema=schema.bind(schema.seed_candidate()),
        task_prompt="extract",
    )
    print(f"    ok={result.ok}  error={result.error}")

    print(
        "\nTo use a real provider, pass any callable:\n"
        "\n"
        "    from anthropic import Anthropic\n"
        "    client = Anthropic()\n"
        "\n"
        "    def llm(prompt: str) -> str:\n"
        "        message = client.messages.create(\n"
        "            model='claude-sonnet-5',\n"
        "            max_tokens=4096,\n"
        "            messages=[{'role': 'user', 'content': prompt}],\n"
        "        )\n"
        "        return message.content[0].text\n"
        "\n"
        "    extractor = TextLayerExtractor(llm)\n"
        "\n"
        "then hand it to optimize_descriptions() exactly as in 02_gemini_invoices.py."
    )


if __name__ == "__main__":
    main()

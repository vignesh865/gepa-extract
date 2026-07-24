"""The full loop, offline and free.

Run:  python -m examples.generate_invoices
      python -m examples.01_offline_invoices

Gold data is real (loaded from the generated corpus). The extraction and
reflection models are stand-ins, so the whole run takes about a second and
needs no API key.

Both stand-ins model the real mechanism rather than faking the outcome:

* ``WeakInvoiceExtractor`` reproduces three failures a small vision model
  really does make on these layouts -- taking the subtotal for the total,
  emitting US-format dates, and inventing a purchase-order number on vendors
  that print none. Each is repaired only when that field's description contains
  a specific instruction, which is exactly the dependency the optimiser exists
  to exploit.
* ``ScriptedReflectionLM`` keys off the *diagnosis* in the feedback it is
  given. If per-field feedback were not reaching it, the run would visibly fail
  to improve.

For real models on these same documents, see 02_gemini_invoices.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from examples.invoice_schema import build_schema
from gepa_extract import ErrorClass, ExtractionResult, load_examples, optimize_descriptions, score_candidate

DATA = Path(__file__).parent / "data" / "invoices.json"

# The instruction each failure needs. The stub extractor keys on these
# substrings; a real extractor simply reads the improved description.
FIX_TOTAL = "beneath the tax line"
FIX_DATE = "iso 8601"
FIX_PO = "only if a purchase order"


class WeakInvoiceExtractor:
    """A deliberately weak extractor, driven by the field descriptions it is given.

    It starts from the gold record and corrupts three fields, undoing each
    corruption when the corresponding description tells it how. That is the same
    contract a real extractor has -- read the description, follow it -- with the
    model replaced by a lookup so the example runs offline.
    """

    def __init__(self, examples) -> None:
        self.gold_by_doc = {example.doc_id: example.gold for example in examples}
        self.calls = 0

    def extract(self, *, document, schema: dict[str, Any], task_prompt: str) -> ExtractionResult:
        self.calls += 1
        gold = self.gold_by_doc.get(document.doc_id)
        if gold is None:
            return ExtractionResult.failure(f"unknown document {document.doc_id!r}")

        data = dict(gold)
        descriptions = {
            field: schema["properties"][field].get("description", "").casefold()
            for field in ("total", "invoice_date", "purchase_order")
        }

        # 1. Location failure: the subtotal sits right next to the total.
        if FIX_TOTAL not in descriptions["total"]:
            data["total"] = gold["subtotal"]

        # 2. Format failure: right day, wrong rendering.
        if FIX_DATE not in descriptions["invoice_date"]:
            year, month, day = gold["invoice_date"].split("-")
            data["invoice_date"] = f"{month}/{day}/{year}"

        # 3. Hallucination: invents a plausible PO on vendors that print none.
        if FIX_PO not in descriptions["purchase_order"] and gold["purchase_order"] is None:
            data["purchase_order"] = f"PO-{document.doc_id[-2:]}00"

        return ExtractionResult.success(data, raw_response="<offline stub>")


class ScriptedReflectionLM:
    """Stands in for the strong vision model.

    A real reflection model would look at the rendered page and write the
    localisation down itself. This one keys off the diagnosis in the feedback,
    which is enough to demonstrate that per-field feedback arrives intact.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.diagnoses: list[str] = []

    def __call__(self, prompt) -> str:
        self.calls += 1
        text = prompt if isinstance(prompt, str) else str(prompt)

        if ErrorClass.SIBLING_VALUE.value in text and "'subtotal'" in text:
            self.diagnoses.append("sibling_value/total")
            reply = (
                "The final amount payable for the whole invoice, including tax. In the summary block it is the "
                f"row {FIX_TOTAL}, and it is the largest of the summary amounts. Do NOT take the amount above "
                "the tax line, which is the subtotal, and do not take a repeated 'balance due' line that follows "
                "the total. Return a number with no currency symbol or thousands separators, e.g. 1320.00."
            )
        elif ErrorClass.FORMAT_MISMATCH.value in text:
            self.diagnoses.append("format_mismatch/invoice_date")
            reply = (
                "The date the invoice was issued, printed near the invoice number. Return it in ISO 8601 format, "
                "YYYY-MM-DD, e.g. 2024-03-14 -- convert from whatever format the document uses."
            )
        elif ErrorClass.HALLUCINATED.value in text:
            self.diagnoses.append("hallucinated/purchase_order")
            reply = (
                "The buyer's purchase order reference. Return it ONLY if a purchase order number is printed on "
                "the document, labelled 'Purchase Order' or 'PO'. Many vendors do not print one; when no such "
                "label appears anywhere on the page, return null. Never substitute the invoice number."
            )
        else:
            # Nothing diagnostic in this batch: leave the description alone.
            reply = _current_description(text)

        return f"```\n{reply}\n```"


def _current_description(prompt_text: str) -> str:
    """Recover the current description from the rendered prompt, to return it unchanged."""
    marker = "Current description for this field:"
    if marker in prompt_text:
        tail = prompt_text.split(marker, 1)[1].strip()
        return tail.split("\n\n", 1)[0].strip()
    return "Unchanged."


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Run `python -m examples.generate_invoices` first.")

    schema = build_schema()
    examples = load_examples(DATA)
    extractor = WeakInvoiceExtractor(examples)

    print(f"{len(examples)} invoices across 3 vendor layouts")
    print(f"{len(schema.field_paths)} fields, each evolved independently\n")

    before, before_fields = score_candidate(schema, extractor, examples, schema.seed_candidate())
    print(f"seed descriptions: {before:.3f} overall")
    for path in ("total", "invoice_date", "purchase_order"):
        print(f"    {path:<16} {before_fields[path]:.3f}")

    reflection_lm = ScriptedReflectionLM()
    result = optimize_descriptions(
        schema,
        extractor,
        trainset=examples,
        reflection_lm=reflection_lm,
        max_metric_calls=400,
        reflection_minibatch_size=4,
        display_progress_bar=False,
        seed=0,
    )

    print(f"\n{result.report()}")
    print(f"\nextraction calls: {extractor.calls}   reflection calls: {reflection_lm.calls}")
    print(f"diagnoses acted on: {sorted(set(reflection_lm.diagnoses))}")

    print("\n" + "-" * 72)
    for path in result.changed_fields:
        print(f"\n[{path}]")
        print(f"  before: {result.seed_descriptions[path]}")
        print(f"  after:  {result.best_descriptions[path]}")


if __name__ == "__main__":
    main()

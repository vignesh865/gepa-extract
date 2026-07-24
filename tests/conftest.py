from __future__ import annotations

import pytest

from gepa_extract import Document, Example, ExtractionSchema

INVOICE_SCHEMA = {
    "type": "object",
    "required": ["invoice_number", "invoice_date", "total", "line_items"],
    "properties": {
        "invoice_number": {"type": "string", "description": "The invoice number."},
        "invoice_date": {"type": "string", "description": "The date of the invoice."},
        "vendor": {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string", "description": "The vendor's name."}},
        },
        "subtotal": {"type": "number", "description": "The subtotal."},
        "tax": {"type": "number", "description": "The tax amount."},
        "total": {"type": "number", "description": "The total amount."},
        "purchase_order": {"type": "string", "description": "The purchase order number."},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["description", "quantity"],
                "properties": {
                    "description": {"type": "string", "description": "The line item description."},
                    "quantity": {"type": "integer", "description": "The quantity."},
                },
            },
        },
    },
}

TASK_PROMPT = "Extract the fields defined by the schema from the attached invoice."
POLICY_TEXT = "Never infer values that are not present in the document."


@pytest.fixture
def schema() -> ExtractionSchema:
    return ExtractionSchema(INVOICE_SCHEMA, task_prompt=TASK_PROMPT, policy_text=POLICY_TEXT)


def make_gold(doc_id: str, *, po: str | None = None) -> dict:
    return {
        "invoice_number": f"INV-{doc_id}",
        "invoice_date": "2024-03-14",
        "vendor": {"name": "Acme Industrial Supply"},
        "subtotal": 1200.00,
        "tax": 120.00,
        "total": 1320.00,
        "purchase_order": po,
        "line_items": [
            {"description": "Steel bracket", "quantity": 4},
            {"description": "Hex bolt, M8", "quantity": 40},
        ],
    }


@pytest.fixture
def examples(tmp_path) -> list[Example]:
    out = []
    for index in range(4):
        doc_id = f"{index:03d}"
        path = tmp_path / f"invoice-{doc_id}.pdf"
        path.write_bytes(b"%PDF-1.4 placeholder")
        out.append(Example(document=Document(doc_id=doc_id, path=path), gold=make_gold(doc_id)))
    return out

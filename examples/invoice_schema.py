"""The invoice schema, with deliberately naive seed descriptions.

These descriptions are what a schema looks like before anyone has tuned it:
short, plausible, and quietly ambiguous. `total` says "the total amount",
which does not distinguish it from the subtotal; `invoice_date` names no
format; `purchase_order` gives the model no way to decide the field is absent.

They are the starting point the optimiser is asked to beat, and they are the
reason it can: every failure the examples demonstrate is latent in this text.
"""

from __future__ import annotations

from gepa_extract import ExtractionSchema

INVOICE_JSON_SCHEMA = {
    "type": "object",
    "required": ["invoice_number", "invoice_date", "vendor", "total", "line_items"],
    "properties": {
        "invoice_number": {"type": "string", "description": "The invoice number."},
        "invoice_date": {"type": "string", "description": "The date of the invoice."},
        "vendor": {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string", "description": "The name of the vendor."}},
        },
        "purchase_order": {"type": "string", "description": "The purchase order number."},
        "subtotal": {"type": "number", "description": "The subtotal."},
        "tax": {"type": "number", "description": "The tax amount."},
        "total": {"type": "number", "description": "The total amount."},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["description", "quantity", "unit_price"],
                "properties": {
                    "description": {"type": "string", "description": "The description of the item."},
                    "quantity": {"type": "integer", "description": "The quantity."},
                    "unit_price": {"type": "number", "description": "The unit price."},
                },
            },
        },
    },
}

TASK_PROMPT = (
    "Extract the fields defined by the JSON schema from the attached invoice. "
    "Follow each field's description exactly."
)

# Frozen alongside the task prompt: present on every extraction call, never part
# of the optimiser's candidate.
POLICY_TEXT = (
    "Only report values that appear in the document. If a field is not present, return null for it. "
    "Do not calculate, infer, or estimate values that are not printed on the page."
)


def build_schema() -> ExtractionSchema:
    return ExtractionSchema(INVOICE_JSON_SCHEMA, task_prompt=TASK_PROMPT, policy_text=POLICY_TEXT)

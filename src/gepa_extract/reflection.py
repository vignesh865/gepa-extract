"""Reflection prompt templates.

GEPA's stock instruction-proposal template asks for a better instruction in
general terms. That is the wrong prompt for this problem, for a specific
reason: **the model reading the description is weaker than the model writing
it.**

The reflection model is a strong VLM that can see the page and reason about
layout. The extraction model is cheaper and typically cannot. So a proposal
like "identify the correct total" is useless -- it asks the weak model to
redo the reasoning the strong model just did. What transfers is the *result*
of that reasoning, written down: positional anchors, neighbouring landmarks,
and explicit negative anchors naming the value to reject.

That framing is the substance of these templates. GEPA accepts a
``{component_name: template}`` dict (``api.py:158``), so each field type can
be steered differently while unlisted fields fall back to the default.

Templates must contain ``<curr_param>`` and ``<side_info>``, and must ask for
the result in a fenced block -- GEPA's output extractor reads the text between
the first and last triple backticks.
"""

from __future__ import annotations

from gepa_extract.schema import ExtractionSchema

__all__ = ["BASE_TEMPLATE", "build_templates"]


_PREAMBLE = """You are improving ONE field's description inside a document-extraction JSON schema.

That description is the entire prompt the extraction model receives for this field. It is read by a
SMALLER, CHEAPER model than you. It cannot see what you can see, and it will not reason its way to
the answer -- it follows instructions literally. Anything you understand about where the value sits
must be written down explicitly, or it is lost.

Current description for this field:

<curr_param>

Below are documents where this field was extracted with the current description, together with the
gold value, the failure class, and images of the pages. Study the images: your advantage over the
extraction model is that you can see the layout, and your job is to convert what you see into text
instructions that do not require sight to follow.

<side_info>
"""

_RULES = """
Write a replacement description. It must:

1. Be a direct instruction to the extraction model about THIS field only. Never mention other fields
   except as values to reject.
2. Prefer RELATIONAL anchors over absolute ones. "In the summary block, the row immediately below the
   tax line" survives a layout change; "in the lower-right corner" does not. Descriptions are scored
   on documents you have not been shown, including unseen vendor templates.
3. Name confusable values explicitly and negatively when the failures show one being taken:
   "...do NOT take the amount immediately above the tax line, which is the subtotal."
4. State the exact output format when format is at issue, with a worked example.
5. State what to return when the field is genuinely absent, and how to recognise that case.
6. Stay under 120 words. A long description dilutes the instruction for a small model.
7. Carry forward whatever in the current description is already working. The failures below are the
   only evidence of what is broken; do not discard the rest.

Do not restate the field's name, type, or whether it is required -- those are fixed by the schema and
are already supplied to the extraction model.

Return ONLY the new description, inside a fenced code block:

```
<the new description>
```
"""

BASE_TEMPLATE = _PREAMBLE + _RULES

_TYPE_NOTES: dict[str, str] = {
    "number": (
        "\nThis field is numeric. Numeric fields fail most often by taking a neighbouring amount from the same "
        "summary block. Anchor it against the surrounding rows and name the amounts that must NOT be taken. "
        "Specify whether currency symbols, thousands separators, and signs belong in the output.\n"
    ),
    "integer": (
        "\nThis field is an integer count. Be explicit about what constitutes one unit, and about rows that wrap "
        "across lines or continue onto a following page.\n"
    ),
    "string": (
        "\nThis field is text. Say whether to copy the document's wording verbatim or normalise it, and how much "
        "surrounding text to include (for example: company name only, excluding any legal suffix or address).\n"
    ),
    "boolean": ("\nThis field is a boolean. Define precisely what evidence in the document makes it true.\n"),
}

_OPTIONAL_NOTE = (
    "\nThis field is OPTIONAL and is absent from some documents. Hallucination is the dominant failure mode for "
    "optional fields: the model finds something plausible and returns it. State the positive evidence required "
    "before returning any value at all, and instruct that null be returned otherwise.\n"
)

_ARRAY_NOTE = (
    "\nThis field is extracted once per row of a repeating table. The confusable values are the OTHER COLUMNS OF "
    "THE SAME ROW. Identify the column by its header and its position relative to the other columns.\n"
)


def build_templates(schema: ExtractionSchema) -> dict[str, str]:
    """Per-field reflection templates, steered by field type and cardinality."""
    templates: dict[str, str] = {}
    for spec in schema.fields:
        notes = _TYPE_NOTES.get(spec.json_type, "")
        if spec.in_array:
            notes += _ARRAY_NOTE
        if not spec.required:
            notes += _OPTIONAL_NOTE
        templates[spec.component_name] = _PREAMBLE + notes + _RULES
    return templates

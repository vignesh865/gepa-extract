"""Error classes for a single extracted field.

A scalar 0/1 tells the reflection model *that* a field was wrong. It does not
tell it *how*, and the fixes diverge sharply: a field that grabbed a
neighbouring value needs positional disambiguation, while a field that
hallucinated needs a stronger absence rule. Classifying the failure is what
turns per-field scoring into per-field instruction.

``guidance`` is written for the reflection model, not for humans -- it is
embedded verbatim in the reflective dataset.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["ErrorClass"]


class ErrorClass(StrEnum):
    CORRECT = "correct"
    FORMAT_MISMATCH = "format_mismatch"
    SIBLING_VALUE = "sibling_value"
    HALLUCINATED = "hallucinated"
    MISSING = "missing"
    WRONG_VALUE = "wrong_value"
    LENGTH_MISMATCH = "length_mismatch"

    @property
    def is_failure(self) -> bool:
        return self is not ErrorClass.CORRECT

    @property
    def guidance(self) -> str:
        """A directive for the reflection model about this failure mode."""
        return _GUIDANCE[self]


_GUIDANCE: dict[ErrorClass, str] = {
    ErrorClass.CORRECT: "",
    ErrorClass.FORMAT_MISMATCH: (
        "The correct value was located, but rendered in the wrong format. The description must state the "
        "required output format explicitly and give a worked example of it. Do not add location guidance -- "
        "locating the value is already working."
    ),
    ErrorClass.SIBLING_VALUE: (
        "A visually or semantically adjacent value was taken instead of the correct one. This is a LOCATION "
        "failure, not a comprehension failure. The description must describe where the correct value sits "
        "relative to stable landmarks in the document, and must explicitly name the confusable value as "
        "something to reject."
    ),
    ErrorClass.HALLUCINATED: (
        "The field is absent from the document, but a value was invented for it. The description must state "
        "the conditions under which the field is genuinely absent and instruct that null be returned, rather "
        "than a plausible-looking substitute."
    ),
    ErrorClass.MISSING: (
        "The value is present in the document but was returned as null. The description must help locate it -- "
        "including where it appears when the layout varies, and any alternative labels it appears under."
    ),
    ErrorClass.WRONG_VALUE: (
        "A value was returned that matches neither the gold value nor any other field. Consider whether the "
        "description is ambiguous about which entity the field refers to."
    ),
    ErrorClass.LENGTH_MISMATCH: (
        "The number of extracted items does not match the document. The description must define what counts as "
        "one item, and how to handle rows that wrap, repeat, or continue across pages."
    ),
}

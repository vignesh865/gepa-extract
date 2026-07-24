"""Value normalisation and comparison.

Extraction is judged against gold values that were typed by humans, so a
comparison of raw values alone is too strict: ``"$1,320.00"`` and ``1320.0``
name the same amount, and ``"01/15/2024"`` and ``"2024-01-15"`` name the same
day. We need to tell those apart from genuinely wrong values *without*
collapsing them together, because "right value, wrong format" and "wrong
value" call for completely different fixes to a field description.

So comparison is three-valued: EXACT, NORMALIZED (equal once formatting is
ignored), DIFFERENT.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

__all__ = ["Match", "compare", "is_absent", "normalize_date", "normalize_number", "normalize_text"]


class Match(StrEnum):
    EXACT = "exact"
    NORMALIZED = "normalized"
    DIFFERENT = "different"


# Ordered by precedence. Day-first and month-first patterns are mutually
# ambiguous for days <= 12 ("03/04/2024"); month-first is tried first because
# the corpora this targets are predominantly US-style invoices. A document set
# that is mostly European should flip _DATE_FORMATS -- there is no way to
# resolve the ambiguity from a single value.
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%m-%d-%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%Y%m%d",
)

_NUMBER_JUNK = re.compile(r"[^0-9.\-]")
_WHITESPACE = re.compile(r"\s+")
_TRAILING_ZEROS = re.compile(r"\.?0+$")


def is_absent(value: Any) -> bool:
    """Treat None, empty string and empty collections as "field not present".

    Extraction models express absence inconsistently -- ``null``, ``""``,
    ``"N/A"`` -- and conflating absence with a wrong value would misclassify
    every optional-field error.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("", "n/a", "na", "none", "null", "-", "--")
    if isinstance(value, (list, dict, tuple)):
        return len(value) == 0
    return False


def normalize_number(value: Any) -> Decimal | None:
    """Parse a currency-ish value to Decimal, or None if it is not numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        return Decimal(str(value))
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    negative = text.startswith("(") and text.endswith(")")  # (1,320.00) accounting style
    # Strip thousands separators before removing junk, so "1.320,50" (European)
    # is not silently read as 1.32050.
    if re.search(r"\d\.\d{3}(?:\D|$)", text) and "," in text:
        text = text.replace(".", "").replace(",", ".")
    else:
        text = text.replace(",", "")

    cleaned = _NUMBER_JUNK.sub("", text)
    if cleaned in ("", "-", ".", "-."):
        return None
    try:
        parsed = Decimal(cleaned)
    except InvalidOperation:
        return None
    return -parsed if negative and parsed > 0 else parsed


def normalize_date(value: Any) -> str | None:
    """Parse a date-ish value to an ISO ``YYYY-MM-DD`` string, or None."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        return None

    text = _WHITESPACE.sub(" ", value.strip())
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            # Invoice dates are calendar dates, not instants. Attaching a
            # timezone would invent information the document does not contain,
            # and .date() discards it anyway.
            return datetime.strptime(text, fmt).date().isoformat()  # noqa: DTZ007
        except ValueError:
            continue
    return None


def normalize_text(value: Any) -> str:
    """Casefold, collapse whitespace, drop surrounding punctuation."""
    text = "" if value is None else str(value)
    text = _WHITESPACE.sub(" ", text).strip().casefold()
    return text.strip(" .,;:'\"")


def compare(extracted: Any, gold: Any) -> Match:
    """Three-valued comparison of one extracted value against its gold value.

    Order matters: numbers are checked before dates because ``"20240115"``
    parses as both, and before text because ``Decimal`` equality is what makes
    ``1320`` == ``1320.00`` true.
    """
    if extracted == gold:
        return Match.EXACT
    if is_absent(extracted) and is_absent(gold):
        return Match.NORMALIZED
    if is_absent(extracted) or is_absent(gold):
        return Match.DIFFERENT

    if isinstance(extracted, bool) or isinstance(gold, bool):
        if _as_bool(extracted) is not None and _as_bool(extracted) == _as_bool(gold):
            return Match.NORMALIZED
        return Match.DIFFERENT

    e_num, g_num = normalize_number(extracted), normalize_number(gold)
    if e_num is not None and g_num is not None:
        return Match.NORMALIZED if e_num == g_num else Match.DIFFERENT

    e_date, g_date = normalize_date(extracted), normalize_date(gold)
    if e_date is not None and g_date is not None:
        return Match.NORMALIZED if e_date == g_date else Match.DIFFERENT

    if normalize_text(extracted) == normalize_text(gold):
        return Match.NORMALIZED
    return Match.DIFFERENT


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in ("true", "yes", "y", "1"):
            return True
        if lowered in ("false", "no", "n", "0"):
            return False
    return None

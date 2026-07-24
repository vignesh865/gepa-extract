"""Generate a real invoice PDF corpus, in three vendor layouts.

Run:  python -m examples.generate_invoices

Writes PDFs plus a gold manifest to ``examples/data/``. Everything is written
with a small pure-Python PDF writer, so the corpus builds with no dependencies
and the files are genuine PDFs that Gemini and pypdfium2 both read.

The three layouts exist to make the *generalisation* problem real rather than
theoretical:

* ``acme``    -- summary block bottom-right, ordered Subtotal / Tax / Total.
* ``globex``  -- summary block on the LEFT, and Total printed ABOVE Subtotal.
* ``initech`` -- Total followed by a "Balance Due" line that repeats the amount,
                 plus a purchase-order number some other vendors omit.

A description that overfits to acme ("the amount in the lower right, beneath
the tax line") scores well on acme and fails on globex. That is exactly the
failure mode worth measuring, which is why ``02_gemini_invoices.py`` holds out
an entire vendor rather than a random sample of documents.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

PAGE_WIDTH, PAGE_HEIGHT = 612, 792


# --------------------------------------------------------------------- PDF writer


class Page:
    """Accumulates text-drawing operators for one page."""

    def __init__(self) -> None:
        self.ops: list[str] = []

    def text(self, x: float, y: float, value: str, *, size: int = 10, bold: bool = False) -> None:
        font = "F2" if bold else "F1"
        escaped = value.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        self.ops.append(f"BT /{font} {size} Tf 1 0 0 1 {x:.2f} {y:.2f} Tm ({escaped}) Tj ET")

    def right_text(self, right_x: float, y: float, value: str, *, size: int = 10, bold: bool = False) -> None:
        # Helvetica averages ~0.5em per character; good enough for right alignment
        # in a synthetic corpus, and keeps the writer dependency-free.
        self.text(right_x - len(value) * size * 0.5, y, value, size=size, bold=bold)

    def line(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.ops.append(f"{x0:.2f} {y0:.2f} m {x1:.2f} {y1:.2f} l S")

    def content(self) -> bytes:
        return "\n".join(self.ops).encode("latin-1")


def write_pdf(path: Path, page: Page) -> None:
    content = page.content()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>"
        ).encode("latin-1"),
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()

    path.write_bytes(bytes(out))


# ------------------------------------------------------------------- corpus data

VENDORS = {
    "acme": "Acme Industrial Supply",
    "globex": "Globex Fabrication Ltd",
    "initech": "Initech Components Inc",
}

ITEM_POOL = [
    ("Steel bracket, 40mm", 4, 62.50),
    ("Hex bolt, M8 x 50", 40, 1.25),
    ("Rubber gasket, 60mm", 12, 8.75),
    ("Aluminium rail, 2m", 3, 145.00),
    ("Cable tie, 200mm (pack of 100)", 6, 11.40),
    ("Bearing housing, type B", 2, 210.00),
]


# Buyers. These appear under "Bill To" and compete with the vendor for
# `vendor.name` -- a document with two company names on it is the single most
# reliable way to break a naive "the name of the vendor" description.
BUYERS = [
    "Northgate Engineering Ltd",
    "Pemberton Facilities Group",
    "Halcyon Marine Services",
    "Westbrook Manufacturing Co",
]


def build_record(vendor_key: str, index: int) -> dict:
    """Deterministic per-document data. No RNG, so the corpus is reproducible.

    Four traps are built in, each targeting a specific error class:

    * ``Bill To`` buyer name competes with the vendor name  -> sibling_value
    * ``Balance Due`` differs from the invoice total        -> sibling_value
    * a discount line, so subtotal + tax != total           -> arithmetic shortcuts fail
    * ``Your reference`` PO label on some vendors only      -> hallucinated / missing

    The amounts are arranged so that every trap value is a *plausible* answer:
    the balance due is a real number on the page, and it is the last amount
    printed, which is where a model looking for "the total" tends to land.
    """
    items = [ITEM_POOL[(index + offset) % len(ITEM_POOL)] for offset in range(2 + index % 3)]
    line_items = [{"description": d, "quantity": q, "unit_price": p} for d, q, p in items]

    subtotal = round(sum(item["quantity"] * item["unit_price"] for item in line_items), 2)
    # A settlement discount breaks the naive subtotal + tax = total arithmetic.
    discount = round(subtotal * 0.05, 2) if index % 2 == 0 else 0.0
    tax = round((subtotal - discount) * 0.10, 2)
    total = round(subtotal - discount + tax, 2)
    # A part-payment already received. `balance_due` is NOT the invoice total,
    # but it is the final amount printed on the page.
    amount_paid = round(total * 0.25, 2) if index % 3 == 0 else 0.0
    balance_due = round(total - amount_paid, 2)

    # acme labels its PO "Your reference", initech labels it "Purchase Order",
    # globex prints none at all.
    purchase_order = None
    if vendor_key == "initech":
        purchase_order = f"PO-{4400 + index}"
    elif vendor_key == "acme" and index % 2 == 0:
        purchase_order = f"{45000 + index}"

    return {
        "doc_id": f"{vendor_key}-{index:02d}",
        "vendor_key": vendor_key,
        "invoice_number": f"{vendor_key[:3].upper()}-2024-{1000 + index}",
        "invoice_date": f"2024-{(index % 12) + 1:02d}-{(index % 27) + 1:02d}",
        "due_date": f"2024-{(index % 12) + 2:02d}-{(index % 27) + 1:02d}",
        "vendor_name": VENDORS[vendor_key],
        "buyer_name": BUYERS[index % len(BUYERS)],
        "purchase_order": purchase_order,
        "line_items": line_items,
        "subtotal": subtotal,
        "discount": discount,
        "tax": tax,
        "total": total,
        "amount_paid": amount_paid,
        "balance_due": balance_due,
    }


def money(value: float) -> str:
    return f"${value:,.2f}"


def render_acme(record: dict) -> Page:
    """Conventional layout: summary bottom-right, Subtotal / Tax / Total."""
    page = Page()
    page.text(56, 730, record["vendor_name"], size=18, bold=True)
    page.text(56, 712, "1400 Foundry Road, Sheffield S9 2XZ")
    page.right_text(556, 730, "INVOICE", size=20, bold=True)
    page.right_text(556, 708, f"Invoice No: {record['invoice_number']}")
    page.right_text(556, 694, f"Date: {record['invoice_date']}")
    page.right_text(556, 680, f"Payment due: {record['due_date']}")
    if record["purchase_order"]:
        page.right_text(556, 666, f"Your reference: {record['purchase_order']}")

    page.text(56, 686, "Bill To:", bold=True)
    page.text(56, 672, record["buyer_name"])

    page.line(56, 655, 556, 655)
    page.text(56, 640, "Description", bold=True)
    page.right_text(400, 640, "Qty", bold=True)
    page.right_text(480, 640, "Unit", bold=True)
    page.right_text(556, 640, "Amount", bold=True)

    y = 620
    for item in record["line_items"]:
        page.text(56, y, item["description"])
        page.right_text(400, y, str(item["quantity"]))
        page.right_text(480, y, money(item["unit_price"]))
        page.right_text(556, y, money(item["quantity"] * item["unit_price"]))
        y -= 18

    y -= 14
    page.line(380, y + 10, 556, y + 10)
    page.text(400, y, "Subtotal")
    page.right_text(556, y, money(record["subtotal"]))
    if record["discount"]:
        page.text(400, y - 16, "Settlement discount")
        page.right_text(556, y - 16, f"-{money(record['discount'])}")
        y -= 16
    page.text(400, y - 16, "Tax (10%)")
    page.right_text(556, y - 16, money(record["tax"]))
    page.text(400, y - 36, "TOTAL", bold=True)
    page.right_text(556, y - 36, money(record["total"]), bold=True)
    if record["amount_paid"]:
        page.text(400, y - 54, "Less payment received")
        page.right_text(556, y - 54, f"-{money(record['amount_paid'])}")
        page.text(400, y - 72, "BALANCE DUE", bold=True)
        page.right_text(556, y - 72, money(record["balance_due"]), bold=True)
    page.text(56, 90, "Payment due within 30 days.")
    return page


def render_globex(record: dict) -> Page:
    """Summary on the LEFT, and Total printed ABOVE Subtotal.

    Any description that says "the amount below the tax line" is wrong here.
    """
    page = Page()
    page.text(56, 740, "INVOICE", size=20, bold=True)
    page.text(56, 716, record["vendor_name"], size=14, bold=True)
    page.text(56, 700, "Unit 7, Parkway Industrial Estate, Leeds LS11 5RD")
    page.right_text(556, 740, record["invoice_number"], size=12, bold=True)
    page.right_text(556, 722, f"Issued {record['invoice_date']}")
    page.right_text(556, 708, f"Due {record['due_date']}")
    page.right_text(556, 690, "Invoice to:", bold=True)
    page.right_text(556, 676, record["buyer_name"])

    page.line(56, 670, 556, 670)
    page.text(56, 650, "Item", bold=True)
    page.right_text(360, 650, "Quantity", bold=True)
    page.right_text(460, 650, "Rate", bold=True)
    page.right_text(556, 650, "Line total", bold=True)

    y = 630
    for item in record["line_items"]:
        page.text(56, y, item["description"])
        page.right_text(360, y, str(item["quantity"]))
        page.right_text(460, y, money(item["unit_price"]))
        page.right_text(556, y, money(item["quantity"] * item["unit_price"]))
        y -= 18

    y -= 30
    page.text(56, y, "AMOUNT PAYABLE", bold=True)
    page.text(190, y, money(record["total"]), bold=True)
    page.text(56, y - 20, "Net goods value")
    page.text(190, y - 20, money(record["subtotal"]))
    if record["discount"]:
        page.text(56, y - 36, "Less settlement discount")
        page.text(190, y - 36, f"-{money(record['discount'])}")
        y -= 16
    page.text(56, y - 36, "VAT at 10%")
    page.text(190, y - 36, money(record["tax"]))
    if record["amount_paid"]:
        page.text(56, y - 56, "Paid on account")
        page.text(190, y - 56, f"-{money(record['amount_paid'])}")
        page.text(56, y - 72, "Remaining balance")
        page.text(190, y - 72, money(record["balance_due"]))
    page.text(56, 90, "Please quote the invoice number when paying.")
    return page


def render_initech(record: dict) -> Page:
    """Total followed by a repeated "Balance Due", plus a PO number."""
    page = Page()
    page.text(56, 736, record["vendor_name"], size=16, bold=True)
    page.text(56, 720, "2200 Technology Park, Austin TX 78727")
    page.right_text(556, 736, "TAX INVOICE", size=16, bold=True)
    page.right_text(556, 716, record["invoice_number"])
    page.right_text(556, 702, record["invoice_date"])
    page.right_text(556, 688, f"Purchase Order: {record['purchase_order']}")
    page.right_text(556, 674, f"Payment due by {record['due_date']}")
    page.text(56, 700, "Sold to:", bold=True)
    page.text(56, 686, record["buyer_name"])

    page.line(56, 664, 556, 664)
    page.text(56, 646, "Part / description", bold=True)
    page.right_text(410, 646, "Units", bold=True)
    page.right_text(490, 646, "Price", bold=True)
    page.right_text(556, 646, "Value", bold=True)

    y = 626
    for item in record["line_items"]:
        page.text(56, y, item["description"])
        page.right_text(410, y, str(item["quantity"]))
        page.right_text(490, y, money(item["unit_price"]))
        page.right_text(556, y, money(item["quantity"] * item["unit_price"]))
        y -= 18

    y -= 16
    page.line(390, y + 10, 556, y + 10)
    page.text(410, y, "Sub-total")
    page.right_text(556, y, money(record["subtotal"]))
    if record["discount"]:
        page.text(410, y - 16, "Discount applied")
        page.right_text(556, y - 16, f"-{money(record['discount'])}")
        y -= 16
    page.text(410, y - 16, "Sales tax")
    page.right_text(556, y - 16, money(record["tax"]))
    page.text(410, y - 34, "Invoice total", bold=True)
    page.right_text(556, y - 34, money(record["total"]), bold=True)
    page.text(410, y - 54, "Credit applied")
    page.right_text(556, y - 54, f"-{money(record['amount_paid'])}")
    page.text(410, y - 70, "Balance due", bold=True)
    page.right_text(556, y - 70, money(record["balance_due"]), bold=True)
    page.text(56, 90, "Remit to account 0042-99183. Late payment incurs 2% monthly interest.")
    return page


RENDERERS = {"acme": render_acme, "globex": render_globex, "initech": render_initech}

# Corporate suffixes stripped by the canonical-vendor convention below.
_SUFFIXES = (" Ltd", " Inc", " LLC", " PLC", " GmbH", " Co")


def gold_for_conventions(record: dict) -> dict:
    """Gold under a set of *organisational conventions* the document never states.

    Current extraction models read these invoices perfectly, so gold that simply
    mirrors the page is scored 1.000 before optimisation begins and demonstrates
    nothing. The failures that survive a capable model are not perception
    failures but definition failures: the model cannot guess which of several
    correct-looking values your organisation means.

    Four conventions, none derivable from the document:

    * ``total``          -- what is *owed now*, i.e. after credits already
                            applied. AP systems book the payable, not the gross.
    * ``subtotal``       -- net of settlement discount, not the printed
                            "Subtotal" line above it.
    * ``vendor.name``    -- canonical registry form, corporate suffix removed.
    * ``purchase_order`` -- digits only, supplier prefixes stripped.

    Every one of these is fixable by writing the rule into the field's
    description, and unfixable by any amount of model capability -- which is
    precisely the regime this tool is for.
    """
    vendor_name = record["vendor_name"]
    for suffix in _SUFFIXES:
        if vendor_name.endswith(suffix):
            vendor_name = vendor_name[: -len(suffix)]
            break

    purchase_order = record["purchase_order"]
    if purchase_order is not None:
        digits = "".join(ch for ch in purchase_order if ch.isdigit())
        purchase_order = digits or None

    return {
        "invoice_number": record["invoice_number"],
        "invoice_date": record["invoice_date"],
        "vendor": {"name": vendor_name},
        "purchase_order": purchase_order,
        "subtotal": round(record["subtotal"] - record["discount"], 2),
        "tax": record["tax"],
        "total": record["balance_due"],
        "line_items": [
            {"description": item["description"], "quantity": item["quantity"], "unit_price": item["unit_price"]}
            for item in record["line_items"]
        ],
    }


def gold_for(record: dict) -> dict:
    """The gold extraction, in the shape of the schema in `invoice_schema.py`."""
    return {
        "invoice_number": record["invoice_number"],
        "invoice_date": record["invoice_date"],
        "vendor": {"name": record["vendor_name"]},
        "purchase_order": record["purchase_order"],
        "subtotal": record["subtotal"],
        "tax": record["tax"],
        "total": record["total"],
        "line_items": [
            {"description": item["description"], "quantity": item["quantity"], "unit_price": item["unit_price"]}
            for item in record["line_items"]
        ],
    }


def main(per_vendor: int = 4) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    literal, conventions = [], []

    for vendor_key in VENDORS:
        for index in range(per_vendor):
            record = build_record(vendor_key, index)
            filename = f"{record['doc_id']}.pdf"
            write_pdf(DATA_DIR / filename, RENDERERS[vendor_key](record))
            base = {"doc_id": record["doc_id"], "file": filename, "vendor": vendor_key}
            literal.append({**base, "gold": gold_for(record)})
            conventions.append({**base, "gold": gold_for_conventions(record)})

    # Two manifests over the same PDFs. Vendors are grouped and ordered so that
    # holding out the tail holds out a whole unseen layout, not a random sample
    # of a layout already seen.
    (DATA_DIR / "invoices.json").write_text(json.dumps(literal, indent=2) + "\n")
    (DATA_DIR / "invoices_conventions.json").write_text(json.dumps(conventions, indent=2) + "\n")

    print(f"wrote {len(literal)} invoices to {DATA_DIR}")
    for vendor_key in VENDORS:
        print(f"  {vendor_key:8} {per_vendor} documents")
    print("\n  invoices.json             gold mirrors the page (current models score ~1.000)")
    print("  invoices_conventions.json gold follows org conventions the page does not state")


if __name__ == "__main__":
    main()

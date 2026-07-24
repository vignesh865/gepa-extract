"""Measuring what a run actually costs, and checking the estimate against it.

Run:  python -m examples.generate_invoices
      python -m examples.05_budget_calibration

``estimate_metric_calls`` is what the README's budget tables are computed from,
so it needs to be right rather than plausible. This sweeps rounds_per_field,
minibatch size and the number of broken fields, runs each configuration for
real, and prints predicted against actual.

The trick that makes the comparison clean is a reflection model that never
improves anything. No proposal is ever accepted, so no candidate is re-evaluated
on the valset, so the ``p*V`` term -- the only estimated part of the formula --
is genuinely zero. What remains is arithmetic, and it should match exactly:

    3V + rounds * unsolved * 2M        (with p = 0)

If a column drifts, the README's budgets are wrong and
``estimate_metric_calls`` needs refitting -- most likely because gepa changed
when it evaluates, not because the corpus changed.

This also serves as a standing check on the coverage guarantee itself: every
configuration asserts that each broken field got its rounds and that no solved
field consumed one.

Offline, no API key, a few seconds.
"""

from __future__ import annotations

import importlib
from typing import Any

from examples.invoice_schema import build_schema
from gepa_extract import ExtractionResult, estimate_metric_calls, load_examples, optimize_descriptions

DATA = importlib.import_module("examples.01_offline_invoices").DATA

ROUNDS = (1, 2, 3)
MINIBATCHES = (2, 4)
BROKEN_COUNTS = (1, 3, 6)


class PermanentlyBrokenExtractor:
    """Corrupts a fixed set of fields and ignores descriptions entirely.

    Deliberately unfixable: a field that got repaired would retire and stop
    consuming rounds, which is the right behaviour in real use but would make
    the cost measurement depend on how quickly the reflection model succeeded.
    Here every round is spent, which is the worst case and the one a budget has
    to cover.
    """

    # Ordered so that BROKEN_COUNTS slices give a mix of types and of
    # required/optional fields.
    BREAKABLE = ("total", "invoice_date", "purchase_order", "subtotal", "tax", "invoice_number")

    def __init__(self, examples: list, n_broken: int) -> None:
        self.gold = {e.doc_id: e.gold for e in examples}
        self.broken = self.BREAKABLE[:n_broken]
        self.calls = 0

    def extract(self, *, document: Any, schema: dict, task_prompt: str) -> ExtractionResult:
        self.calls += 1
        data = dict(self.gold[document.doc_id])
        for path in self.broken:
            current = data.get(path)
            # Wrong in a type-appropriate way, so scoring classifies rather than
            # discarding the value.
            data[path] = -1.0 if isinstance(current, (int, float)) and not isinstance(current, bool) else "XX-WRONG"
        return ExtractionResult.success(data)


class NeverImprovesReflectionLM:
    """Returns a valid but useless description, so nothing is ever accepted."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, prompt: Any) -> str:
        self.calls += 1
        return "```\nA replacement description that changes nothing.\n```"


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Run `python -m examples.generate_invoices` first.")

    schema = build_schema()
    examples = load_examples(DATA)
    n_val = len(examples)

    print(f"corpus: {n_val} documents, {len(schema.field_paths)} fields")
    print(f"formula: 3V + rounds x unsolved x (2M + p*V), with V={n_val}, p=0\n")

    header = (
        f"{'broken':>6} {'rounds':>6} {'M':>3} | {'actual':>7} {'predicted':>9} | "
        f"{'post-run':>8} {'reflections':>11}  match"
    )
    print(header)
    print("-" * len(header))

    mismatches = 0
    for n_broken in BROKEN_COUNTS:
        for rounds in ROUNDS:
            for minibatch in MINIBATCHES:
                extractor = PermanentlyBrokenExtractor(examples, n_broken)
                reflection_lm = NeverImprovesReflectionLM()
                result = optimize_descriptions(
                    schema,
                    extractor,
                    trainset=examples,
                    reflection_lm=reflection_lm,
                    rounds_per_field=rounds,
                    max_metric_calls=None,
                    reflection_minibatch_size=minibatch,
                    seed=0,
                )

                # The guarantee, checked on every configuration.
                for path in extractor.broken:
                    assert result.rounds[path] >= rounds, f"{path} short of its {rounds} rounds"
                served = {p for p, n in result.rounds.items() if n}
                assert served == set(extractor.broken), f"turns went to solved fields: {served - set(extractor.broken)}"

                predicted = estimate_metric_calls(
                    n_broken,
                    n_val,
                    rounds_per_field=rounds,
                    reflection_minibatch_size=minibatch,
                    acceptance_rate=0.0,
                )
                ok = predicted == extractor.calls
                mismatches += not ok
                print(
                    f"{n_broken:>6} {rounds:>6} {minibatch:>3} | {extractor.calls:>7} {predicted:>9} | "
                    f"{2 * n_val:>8} {reflection_lm.calls:>11}  {'yes' if ok else 'NO'}"
                )

    total = len(BROKEN_COUNTS) * len(ROUNDS) * len(MINIBATCHES)
    print(f"\n{total - mismatches}/{total} configurations matched the estimate exactly.")
    if mismatches:
        raise SystemExit("estimate_metric_calls no longer matches measured cost; the README budgets are stale.")

    print(
        "\nNote the post-run column: optimize_descriptions scores the seed and the best\n"
        "candidate after the run, two passes over the evaluation set that sit outside\n"
        "gepa's budget. On the cheapest configuration above they are most of the bill."
    )


if __name__ == "__main__":
    main()

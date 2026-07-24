"""What a metric-call budget does not buy you: coverage.

Run:  python -m examples.generate_invoices
      python -m examples.04_coverage_budget

``max_metric_calls`` is denominated in document extractions. It answers "what
does this run cost" and says nothing about which fields get a turn -- and the
two come apart badly. This example runs the *same* corpus, extractor and
reflection model twice, changing only the stopping condition, and counts where
the reflection turns actually went.

Run A uses gepa's stock ``round_robin``, wrapped in a counter. Round-robin is
blind to whether a field needs help: it cycles through all ten in order,
spending reflection on the seven that already score 1.000.

Run B uses ``rounds_per_field``, which serves the least-served field that is
not yet solved, and stops once every such field has had its turns.

Both reach the same score. The difference is what they spent getting there.

The stand-ins are imported from 01_offline_invoices, so this is the same loop
you have already seen -- no API key, about a second.
"""

from __future__ import annotations

import importlib
import json
from typing import Any

from gepa.strategies.component_selector import RoundRobinReflectionComponentSelector

from examples.invoice_schema import build_schema
from gepa_extract import load_examples, optimize_descriptions, score_candidate

# 01_offline_invoices starts with a digit, so it cannot be imported by name.
_offline = importlib.import_module("examples.01_offline_invoices")
WeakInvoiceExtractor = _offline.WeakInvoiceExtractor
ScriptedReflectionLM = _offline.ScriptedReflectionLM
DATA = _offline.DATA


class CountingRoundRobin:
    """gepa's stock selector, instrumented.

    Delegates every decision to ``RoundRobinReflectionComponentSelector`` -- the
    point is to observe the real strategy, not to model it -- and tallies which
    field each turn went to.
    """

    def __init__(self) -> None:
        self._inner = RoundRobinReflectionComponentSelector()
        self.picks: list[str] = []

    def __call__(self, state: Any, trajectories: Any, scores: Any, candidate_idx: int, candidate: dict) -> list[str]:
        chosen = self._inner(state, trajectories, scores, candidate_idx, candidate)
        self.picks.extend(chosen)
        return chosen


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Run `python -m examples.generate_invoices` first.")

    schema = build_schema()
    examples = load_examples(DATA)

    # Which fields are actually broken before anything is optimised. Turns spent
    # on any other field are turns spent on a field already scoring 1.000.
    _, seed_fields = score_candidate(schema, WeakInvoiceExtractor(examples), examples, schema.seed_candidate())
    broken = sorted(path for path, score in seed_fields.items() if score < 1.0)

    print(f"{len(schema.field_paths)} fields, of which {len(broken)} need work: {', '.join(broken)}\n")

    # -- Run A: a metric-call budget, with stock round-robin selection ---------
    selector = CountingRoundRobin()
    extractor_a = WeakInvoiceExtractor(examples)
    result_a = optimize_descriptions(
        schema,
        extractor_a,
        trainset=examples,
        reflection_lm=ScriptedReflectionLM(),
        max_metric_calls=400,
        module_selector=selector,
        reflection_minibatch_size=4,
        seed=0,
    )
    wasted = [path for path in selector.picks if path not in broken]

    print("A. max_metric_calls=400, stock round_robin")
    print(f"     score            {result_a.seed_score:.3f} -> {result_a.best_score:.3f}")
    print(f"     extractions      {extractor_a.calls}")
    print(f"     reflection turns {len(selector.picks)}")
    print(f"     ...of which went to fields already at 1.000: {len(wasted)}")

    # -- Run B: a coverage budget --------------------------------------------
    extractor_b = WeakInvoiceExtractor(examples)
    result_b = optimize_descriptions(
        schema,
        extractor_b,
        trainset=examples,
        reflection_lm=ScriptedReflectionLM(),
        rounds_per_field=2,
        max_metric_calls=None,  # coverage is the only stopping condition
        reflection_minibatch_size=4,
        seed=0,
    )
    turns = sum(result_b.rounds.values())
    wasted_b = sum(n for path, n in result_b.rounds.items() if path not in broken)

    print("\nB. rounds_per_field=2")
    print(f"     score            {result_b.seed_score:.3f} -> {result_b.best_score:.3f}")
    print(f"     extractions      {extractor_b.calls}")
    print(f"     reflection turns {turns}")
    print(f"     ...of which went to fields already at 1.000: {wasted_b}")

    print("\nrounds per field:")
    for path in sorted(result_b.rounds, key=lambda p: (-result_b.rounds[p], p)):
        mark = "needs work" if path in broken else "already 1.000"
        print(f"  {path:<28} {result_b.rounds[path]}   ({mark})")

    # Every broken field was served; and because each was fixed on its first
    # proposal, it retired rather than claiming the second round it was owed.
    # "2 rounds each" is a ceiling on waste, not a quota to be spent.
    print(f"\nnever given a round: {len(result_b.unvisited_fields)}/{len(schema.field_paths)}")
    print("every field needing work was served:", all(result_b.rounds[p] >= 1 for p in broken))

    # -- What the run actually produced --------------------------------------
    # Two views of the same result. The per-field descriptions are what the
    # optimiser evolved; the assembled schema is what you would ship, and is
    # the only form the extraction model ever sees.
    print("\n" + "=" * 72)
    print("FINAL DESCRIPTIONS (individual)")
    print("=" * 72)
    for path in schema.field_paths:
        changed = path in result_b.changed_fields
        print(f"\n[{path}]{'  <- rewritten' if changed else ''}")
        if changed:
            print(f"  before: {result_b.seed_descriptions[path]}")
            print(f"  after:  {result_b.best_descriptions[path]}")
        else:
            print(f"  {result_b.best_descriptions[path]}")

    print("\n" + "=" * 72)
    print("FINAL SCHEMA (assembled)")
    print("=" * 72)
    # Drops the evolved descriptions into the frozen skeleton. Keys, types,
    # nesting and required-ness come from the skeleton the optimiser never saw,
    # which is why the fingerprint below cannot have moved.
    assembled = result_b.assembled_schema(schema)
    print(json.dumps(assembled, indent=2))
    print(
        "\nstructure identical to the input schema:",
        schema.structural_fingerprint_of(assembled) == schema.fingerprint(),
    )


if __name__ == "__main__":
    main()

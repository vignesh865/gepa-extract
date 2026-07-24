"""A real optimisation run: Gemini extraction, Gemini vision reflection.

Run:  pip install -e '.[gemini,render]'
      export GOOGLE_API_KEY=...
      python -m examples.generate_invoices
      python -m examples.02_gemini_invoices

This costs money. ``--budget`` counts document extractions, and every one is a
real vision request; start small and raise it once the run looks sane.

Two choices here are the substance of the example:

**The models are deliberately asymmetric.** Extraction runs on a small model
that reads the descriptions; reflection runs on a large vision model that
writes them. Optimising against a strong extractor teaches you nothing about
whether a description carries enough information for a weak one -- and the weak
one is what you deploy.

**The holdout is a whole unseen vendor.** ``initech`` is never trained on. A
random document-level split would overstate accuracy badly here, because a
description that memorises acme's layout scores well on another acme invoice.
The question worth answering is whether the evolved descriptions transfer to a
layout the optimiser never saw.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from examples.invoice_schema import build_schema
from gepa_extract import load_examples, optimize_descriptions, score_candidate
from gepa_extract.backends import GeminiExtractor, GeminiReflectionLM
from gepa_extract.rendering import PdfiumRenderer

DATA_DIR = Path(__file__).parent / "data"
CACHE = Path(__file__).parent / ".cache" / "pages"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=200, help="max document extractions (default: 200)")
    parser.add_argument("--extraction-model", default="gemini-2.5-flash", help="the weak model that reads")
    parser.add_argument("--reflection-model", default="gemini-2.5-pro", help="the strong VLM that writes")
    parser.add_argument("--minibatch", type=int, default=4)
    parser.add_argument("--max-image-docs", type=int, default=3, help="documents per reflection carrying page images")
    parser.add_argument("--run-dir", default=None, help="where GEPA writes checkpoints and best outputs")
    parser.add_argument("--no-images", action="store_true", help="text-only reflection, to measure what vision buys")
    parser.add_argument(
        "--manifest",
        default="invoices_conventions.json",
        help=(
            "gold manifest. 'invoices.json' mirrors the page, which current models already score ~1.000 -- "
            "leaving nothing to optimise. The default encodes organisational conventions the document does "
            "not state, which is where description quality still decides the answer."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = DATA_DIR / args.manifest
    if not manifest.exists():
        raise SystemExit("Run `python -m examples.generate_invoices` first.")

    schema = build_schema()
    examples = load_examples(manifest)

    # Hold out an entire vendor layout, not a random sample of documents.
    train = [e for e in examples if not e.doc_id.startswith("initech")]
    holdout = [e for e in examples if e.doc_id.startswith("initech")]

    extractor = GeminiExtractor(args.extraction_model)
    renderer = None if args.no_images else PdfiumRenderer(cache_dir=CACHE)

    print(f"gold: {args.manifest}")
    print(f"train: {len(train)} documents (acme, globex)")
    print(f"holdout: {len(holdout)} documents (initech -- never seen during optimisation)")
    print(f"extraction: {args.extraction_model}   reflection: {args.reflection_model}")
    print(f"reflection images: {'off' if args.no_images else 'on'}   budget: {args.budget} extractions\n")

    seed = schema.seed_candidate()
    holdout_before, holdout_fields_before = score_candidate(schema, extractor, holdout, seed)
    print(f"holdout with seed descriptions: {holdout_before:.3f}\n")

    result = optimize_descriptions(
        schema,
        extractor,
        trainset=train,
        reflection_lm=GeminiReflectionLM(args.reflection_model),
        renderer=renderer,
        max_metric_calls=args.budget,
        reflection_minibatch_size=args.minibatch,
        max_image_documents=args.max_image_docs,
        run_dir=args.run_dir,
        seed=0,
    )

    print("\n=== training set " + "=" * 55)
    print(result.report())

    print("\n=== holdout: unseen vendor layout " + "=" * 38)
    holdout_after, holdout_fields_after = score_candidate(schema, extractor, holdout, result.best_descriptions)
    print(f"overall   {holdout_before:.3f} -> {holdout_after:.3f}  ({holdout_after - holdout_before:+.3f})\n")
    for path in sorted(holdout_fields_after):
        before, after = holdout_fields_before[path], holdout_fields_after[path]
        if before != after:
            marker = "+" if after > before else "-"
            print(f"  {marker} {path:<28} {before:.3f} -> {after:.3f}")

    # A field that improves on training but drops on the holdout has learned a
    # layout cue rather than a rule. That is the failure worth catching, and it
    # is invisible without a vendor-level holdout.
    overfitted = [
        path
        for path in result.changed_fields
        if result.per_field_best[path] > result.per_field_seed[path]
        and holdout_fields_after[path] < holdout_fields_before[path]
    ]
    if overfitted:
        print(f"\n! improved on training but regressed on the unseen layout: {overfitted}")
        print("  These descriptions likely encode an acme/globex-specific cue.")

    print("\n=== evolved descriptions " + "=" * 47)
    for path in result.changed_fields:
        print(f"\n[{path}]")
        print(f"  before: {result.seed_descriptions[path]}")
        print(f"  after:  {result.best_descriptions[path]}")


if __name__ == "__main__":
    main()

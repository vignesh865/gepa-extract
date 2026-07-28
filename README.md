
# gepa-extract

Document extraction runs on a JSON schema in which each field's `description`
is really a per-field prompt. Those descriptions are usually hand-tuned by
guesswork, and that guesswork is where accuracy leaks: `total` grabs the
subtotal, dates come back in the wrong format, optional fields get hallucinated
when they are absent from the document.

`gepa-extract` optimises those descriptions automatically, using gold
extractions as the signal. Each field is a separately-evolved
[GEPA](https://github.com/gepa-ai/gepa) component.

```python
from gepa_extract import ExtractionSchema, load_examples, optimize_descriptions
from gepa_extract.backends import GeminiExtractor, GeminiReflectionLM
from gepa_extract.rendering import PdfiumRenderer

schema = ExtractionSchema(
    invoice_json_schema,
    task_prompt="Extract the fields defined by the schema from the attached invoice.",
    policy_text="Never infer values that are not present in the document.",
)

result = optimize_descriptions(
    schema,
    extractor=GeminiExtractor("gemini-2.5-flash"),      # weak: reads the descriptions
    trainset=load_examples("data/train.json"),
    reflection_lm=GeminiReflectionLM("gemini-2.5-pro"),  # strong: writes them
    renderer=PdfiumRenderer(cache_dir=".cache/pages"),
    max_metric_calls=300,
)

print(result.report())
print(result.best_descriptions["total"])

# The shippable artifact: evolved descriptions dropped back into the frozen
# skeleton. This is what you hand an extractor.
optimised_schema = result.assembled_schema(schema)
```

## Why GEPA fits

Comparing an extraction against gold gives *per-field* feedback — "6/10
documents: `total` took the value belonging to `subtotal`" — which is far
sharper than a single global score. GEPA consumes exactly that: each field is
a component with its own reflective dataset, and per-field scores become
objectives on a Pareto front, so a candidate that is uniquely good at one field
survives even when its mean is unremarkable.

## Budgets: cost and coverage are different questions

`max_metric_calls` is denominated in *document extractions*. It answers "what
does this run cost", and says nothing about how many fields ever get a turn —
on a wide schema, most of them don't. With 100 fields, a 20-document valset and
a 300-call budget, roughly a dozen fields are reached and the other ~90 keep
their seed descriptions. Nothing reports this.

`rounds_per_field` answers the other question directly:

```python
result = optimize_descriptions(
    schema, extractor, trainset,
    reflection_lm=...,
    rounds_per_field=2,     # every field that still fails gets 2 attempts
    max_metric_calls=None,  # or keep it as a cost ceiling
)
print(result.unvisited_fields)   # fields that never got a round
```

A *round* is one selection of a field for reflection. `FieldCoverageSelector`
keeps a single global round count and always serves the least-served field that
has not yet scored 1.0, so fields that get fixed retire and hand their turns to
fields still failing — the cost tracks how many fields are actually broken, not
how many exist.

This replaces stock `round_robin`, which is blind on two counts. It cycles
through every field in order whether or not that field needs help, and its
cursor is stored **per parent candidate** — so as the Pareto front grows the
population walks several independent cursors and coverage fragments, some
fields being proposed for repeatedly on one lineage while others are never
reached on any.

`examples/04_coverage_budget.py` measures the gap on the ten-field invoice
schema, where only three fields are actually broken:

```
A. max_metric_calls=400, stock round_robin   0.800 -> 1.000
     extractions 424    reflection turns 7  (4 spent on fields already at 1.000)
B. rounds_per_field=2                        0.800 -> 1.000
     extractions  84    reflection turns 3  (0 spent on fields already at 1.000)
```

The guarantee is deliberately narrow — every unsolved field is *selected* at
least N times. Not that the proposals are good, and not that they are accepted.
It is also denominated in rounds rather than iterations, because gepa abandons
an iteration before consulting the selector when every sampled score is already
perfect; an iteration-counted budget would quietly weaken exactly as fields
start passing. `max_iterations` remains as a backstop, and hitting it means the
guarantee was *not* met — which is what `unvisited_fields` is for.

## Choosing `rounds_per_field`

### Ask the planner

`plan_budget` turns "I have this many fields and this many documents" into
settings, with the reasoning attached:

```python
from gepa_extract import plan_budget, score_candidate

# The one input worth measuring rather than guessing: cost scales with the
# fields that are actually broken, not with how many the schema has.
_, per_field = score_candidate(schema, extractor, docs, schema.seed_candidate())
unsolved = sum(1 for s in per_field.values() if s < 1.0)

print(plan_budget(n_fields=120, n_documents=40, n_unsolved_fields=unsolved).explain())
```

```
rounds_per_field           2
reflection_minibatch_size  5
max_metric_calls           1201
valset / holdout           32 / 8

estimated cost             ~801 document extractions
                           ~36 reflection calls
```

Pass `max_extractions` if you have a ceiling and it will solve for rounds
rather than assuming them, or tell you plainly that nothing fits:

```
No affordable plan: the ceiling does not cover even one round.

cheapest possible run       ~2448 document extractions
                            (1 round over 120 fields, valset 32, minibatch 5)

- Assuming all 120 fields need work, which is a ceiling. Score the seed
  candidate first and pass n_unsolved_fields -- it is usually far smaller,
  and cost scales with it.
- Even one round costs ~2448, above the 500 ceiling. [...]
```

That is the same schema and the same 500-call ceiling as the run above: measuring
`unsolved` first is what turns "nothing fits" into an affordable one-round plan.

The advice is a defensible starting point, not a tuned value. The rest of this
section is what it is reasoning from, if you would rather set the numbers
yourself.

### What a run costs

```
3V  +  rounds × unsolved × (2M + p·V)
```

`V` = valset size, `M` = `reflection_minibatch_size`, `p` = fraction of
proposals accepted, `unsolved` = fields scoring below 1.0 on the seed.

| Term | Why |
| --- | --- |
| `3V` | GEPA's seed evaluation, **plus the two passes `optimize_descriptions` makes after the run** to report seed and best scores. Those two sit outside GEPA's budget and are easy to forget — on a short run they dominate everything else. |
| `2M` per round | GEPA evaluates the parent, then the child, on that round's minibatch. |
| `p·V` per round | An accepted child is re-evaluated on the whole valset. |

Only `p` is an estimate; the rest is exact. Checked against 18 real runs (1–6
unsolved fields × 1–3 rounds × minibatch 2 and 4) driven by a reflection model
that never improves anything — so nothing is accepted, `p` is genuinely zero,
and what remains is arithmetic. All 18 matched to the extraction.
`tests/test_selectors.py` pins four of them, so if a gepa upgrade changes when
it evaluates, these budgets fail loudly instead of going quietly stale.

### The number that drives cost is `unsolved`, not your field count

Solved fields retire without consuming rounds, so a 100-field schema with 12 bad
fields costs what 12 costs — not what 100 costs. Measure it before budgeting:

```python
_, per_field = score_candidate(schema, extractor, valset, schema.seed_candidate())
unsolved = [f for f, s in per_field.items() if s < 1.0]

estimate_metric_calls(len(unsolved), len(valset), rounds_per_field=2)
```

That first call costs one pass over the valset and is the best-spent budget in
the whole run: it tells you both what to pay for and what the optimiser is
actually being asked to fix.

### Extractions, at `V=20`, `M=5`, `p=0.3`

| unsolved fields | 1 round | 2 rounds | 3 rounds |
| ---: | ---: | ---: | ---: |
| 5 | 140 | 220 | 300 |
| 10 | 220 | 380 | 540 |
| 25 | 460 | 860 | 1,260 |
| 50 | 860 | 1,660 | 2,460 |
| 100 | 1,660 | 3,260 | 4,860 |
| 250 | 4,060 | 8,060 | 12,060 |
| 500 | 8,060 | 16,060 | 24,060 |

### How to set it

**Start at `rounds_per_field=2`.** One round gives every field a single attempt
with no recovery from a bad proposal; two lets a field that regressed or was
misdiagnosed be revisited. Past three, returns fall off sharply — a description
that has not improved in three attempts usually has a problem reflection cannot
see, most often a gold value that is itself wrong.

**Then use the cheaper levers before cutting rounds.** Both scale the whole
run, and neither touches the coverage guarantee:

| Lever | Effect on 100 unsolved fields, 2 rounds |
| --- | --- |
| `V=20`, `M=5` (defaults) | 3,260 |
| `M=3` | 2,460 |
| `V=10`, `M=3` | 1,830 |
| `V=5`, `M=3` | 1,515 |

A small valset is the strongest lever, and the one to be careful with: it is
also your only estimate of whether a description generalises. Below ~8
documents a single unusual invoice starts steering the whole run. Prefer
cutting `M` first.

**Worked example — 120-field schema, Gemini Flash.** Seed scoring finds 18
fields below 1.0. At `V=15`, `M=3`, `rounds_per_field=2`:

```
3(15) + 2 × 18 × (2·3 + 0.3·15) = 45 + 36 × 10.5 = 423 extractions
```

Roughly 423 documents through Flash and ~36 reflection calls through Pro —
against 120 fields that would have been unaffordable to cover naively. Budget
the same run by metric calls instead and you would have to guess, and would
almost certainly leave most of the 18 untouched.

Keep `max_metric_calls` set as well. It costs nothing when the estimate holds
and caps the damage when `p` turns out much higher than assumed:

```python
optimize_descriptions(
    schema, extractor, trainset,
    reflection_lm=...,
    rounds_per_field=2,
    max_metric_calls=800,   # ceiling, ~2x the estimate
)
```

Then check `result.unvisited_fields` — if it contains fields scoring below 1.0,
the ceiling stopped the run before coverage completed and the guarantee did not
hold.

## Only descriptions evolve

Keys, types, nesting, required-ness, the task prompt and any organisational
policy text are **structurally unreachable** by the optimiser. This is enforced
by construction, not by validation: the candidate GEPA mutates is a flat
`{field_path: description}` mapping, and the schema is reassembled at
evaluation time by dropping those descriptions into a skeleton the optimiser
never sees. There is no code path through which a mutation could alter
structure. `ExtractionSchema.fingerprint()` exists to prove that in a test.

## The capability asymmetry

The model that *reads* a description is smaller and cheaper than the model that
*writes* it. The reflection model is a strong VLM that sees the rendered pages
and can reason about layout; the extraction model typically cannot. So a
proposal like "identify the correct total" is worthless — it asks the weak
model to redo reasoning it cannot do.

What transfers is the *result* of that reasoning, written down:

> The grand total appears in the summary block, in the final row **beneath**
> the tax line. The visually similar amount immediately **above** the tax line
> is the subtotal — do not take it.

The optimiser is therefore a capability-distillation loop: strong-model
perception, compiled into text a weak model can follow. The reflection prompts
in `reflection.py` are written to push for relational anchors ("below the tax
line") over absolute ones ("lower right"), because relational cues survive a
change of vendor template and absolute ones do not.

## Error classes

Scoring classifies *how* each field failed, because the fixes diverge:

| Class | Meaning | What the description needs |
| --- | --- | --- |
| `sibling_value` | Returned another field's gold value | Positional disambiguation + a negative anchor |
| `format_mismatch` | Right value, wrong rendering | An explicit format with a worked example |
| `hallucinated` | Field absent, value invented | Absence conditions, instruction to return null |
| `missing` | Field present, null returned | Location help, alternative labels |
| `length_mismatch` | Wrong number of rows | A definition of what counts as one item |
| `wrong_value` | None of the above | Disambiguation of which entity is meant |

`sibling_value` needs no document coordinates: if the extracted value is
*exactly some other field's gold value*, that is detectable from the data alone,
and it is the difference between "wrong" and "took the subtotal".

## Repeating tables are matched by content, not position

Rows of a `line_items[]` table are paired with gold rows by similarity, not by
index. Comparing row *i* against gold row *i* means one reordered or dropped row
misaligns everything below it, and the damage is not just a low score — comparing
row 2 against row 1's gold routinely lands on another row's value and trips
sibling detection, so the reflection model is told it confused two columns when
it actually read the right cell of the wrong row, and it writes positional
anchors into a description that was already correct.

Alignment is computed once per table from *all* of its columns jointly, then
shared by each column's scoring — per-column alignment would let `quantity` and
`description` choose different row orders, destroying the same-row scoping that
makes sibling detection mean anything. Every column votes, so a column that is
broken everywhere still gets its rows matched by the columns that are not.
Unmatched rows are a count disagreement and stay `length_mismatch`; a dropped
row now costs one row instead of the whole table.

Order is never scored. No description of `line_items[].quantity` controls the
order rows come back in, so penalising it feeds the optimiser noise it cannot
act on. Pass `array_order="positional"` if document order is itself part of your
contract.

## Documents

Extraction receives the **native PDF** — Gemini reads it directly, text layer
intact. Reflection receives **rasterised pages**, because GEPA can carry images
into a reflection prompt but has no document content part (verified against
gepa 0.1.4; `Image(path="x.pdf")` does not error, it silently mislabels the PDF
as `image/png`). So `rendering.py` always passes `media_type` explicitly rather
than letting it be inferred.

Whole pages are sent, never crops: extraction returns values, not coordinates,
so there is no region to crop to — and cropping would defeat the purpose of
letting a strong model do the localisation.

## Install

```bash
pip install -e '.[gemini,render,dev]'
export GOOGLE_API_KEY=...
```

Extras: `gemini` (google-genai), `render` (pypdfium2 + Pillow), `dev` (pytest).
The core installs with neither, and the whole test suite runs offline against
`StubExtractor`.

## Examples

| Example | Needs an API key | What it shows |
| --- | --- | --- |
| `examples/01_offline_invoices.py` | no | The full loop end to end, including a `total`→subtotal failure being diagnosed and fixed |
| `examples/02_gemini_invoices.py` | yes | Real Gemini extraction over real PDFs, vision reflection, holdout scoring |
| `examples/03_custom_backend.py` | no | Adding a provider by implementing `Extractor` |
| `examples/04_coverage_budget.py` | no | The same run under a metric-call budget vs `rounds_per_field` — identical score, 424 extractions vs 84 |
| `examples/generate_invoices.py` | no | Builds the PDF corpus used above, in three vendor layouts |

Run them as modules, so the shared schema import resolves:

```bash
python -m examples.generate_invoices    # writes 12 real PDFs + gold manifest
python -m examples.01_offline_invoices  # ~1s, no API key
```

The offline example takes the seed descriptions from 0.800 to 1.000 and changes
exactly three of ten descriptions — the three it was given evidence about.

## Adding a backend

Implement one method. Nothing about GEPA, scoring, or optimisation crosses this
boundary:

```python
class MyExtractor:
    def extract(self, *, document, schema, task_prompt) -> ExtractionResult:
        ...
```

Return `ExtractionResult.failure(...)` rather than raising: one malformed
response is data about the current candidate, not a reason to abort a run that
may be hours in.

## Layout

```
schema.py      frozen skeleton + candidate binding   <- the safety property
scoring.py     per-field scoring, sibling detection,
               content-based row alignment            <- the signal
errors.py      error classes + reflection guidance
reflection.py  asymmetry-aware prompt templates
selectors.py   coverage guarantee over fields          <- rounds, not calls
rendering.py   PDF -> PNG for the reflection model
extraction.py  the backend interface
adapter.py     the GEPA seam  ) the only two modules
optimize.py    the runner     ) that import gepa
```

That seam is deliberate. If GEPA's pre-1.0 API moves, or a different search
engine is ever wanted, the domain layer ports unchanged.
`tests/test_gepa_contracts.py` pins the library behaviours this depends on, so
an upgrade that breaks them fails loudly instead of silently degrading.

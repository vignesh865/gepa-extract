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
```

## Why GEPA fits

Comparing an extraction against gold gives *per-field* feedback — "6/10
documents: `total` took the value belonging to `subtotal`" — which is far
sharper than a single global score. GEPA consumes exactly that: each field is
a component with its own reflective dataset, and per-field scores become
objectives on a Pareto front, so a candidate that is uniquely good at one field
survives even when its mean is unremarkable.

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
| `examples/generate_invoices.py` | no | Builds the PDF corpus used above, in three vendor layouts |

Run them as modules, so the shared schema import resolves:

```bash
python -m examples.generate_invoices    # writes 12 real PDFs + gold manifest
python -m examples.01_offline_invoices  # ~1s, no API key
```

The offline example takes the seed descriptions from 0.783 to 1.000 and changes
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
scoring.py     per-field scoring + sibling detection <- the signal
errors.py      error classes + reflection guidance
reflection.py  asymmetry-aware prompt templates
rendering.py   PDF -> PNG for the reflection model
extraction.py  the backend interface
adapter.py     the GEPA seam  ) the only two modules
optimize.py    the runner     ) that import gepa
```

That seam is deliberate. If GEPA's pre-1.0 API moves, or a different search
engine is ever wanted, the domain layer ports unchanged.
`tests/test_gepa_contracts.py` pins the library behaviours this depends on, so
an upgrade that breaks them fails loudly instead of silently degrading.

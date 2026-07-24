"""gepa-extract: evolve document-extraction field descriptions against gold data.

A JSON extraction schema's field descriptions are per-field prompts, usually
hand-tuned by guesswork. This package treats each one as a separately-evolved
GEPA component, scores extractions field by field against gold, and lets a
strong vision model rewrite the descriptions a weaker extraction model reads.

Everything except the descriptions -- keys, types, nesting, the task prompt,
organisational policy text -- is outside the optimiser's candidate and so
cannot be modified by it.
"""

from gepa_extract.adapter import ExtractionAdapter
from gepa_extract.documents import Document, Example, load_examples, split
from gepa_extract.errors import ErrorClass
from gepa_extract.extraction import CallableExtractor, ExtractionResult, Extractor, StubExtractor
from gepa_extract.optimize import (
    OptimizationResult,
    estimate_metric_calls,
    optimize_descriptions,
    score_candidate,
)
from gepa_extract.reflection import build_templates
from gepa_extract.rendering import NullRenderer, PageRenderer, PdfiumRenderer, RenderedPage, StubRenderer
from gepa_extract.schema import ExtractionSchema, FieldSpec
from gepa_extract.scoring import DocumentOutcome, FieldOutcome, score_document
from gepa_extract.selectors import FieldCoverageSelector, RoundsPerFieldStopper

__all__ = [
    "CallableExtractor",
    "Document",
    "DocumentOutcome",
    "ErrorClass",
    "Example",
    "ExtractionAdapter",
    "ExtractionResult",
    "ExtractionSchema",
    "Extractor",
    "FieldCoverageSelector",
    "FieldOutcome",
    "FieldSpec",
    "NullRenderer",
    "OptimizationResult",
    "PageRenderer",
    "PdfiumRenderer",
    "RenderedPage",
    "RoundsPerFieldStopper",
    "StubExtractor",
    "StubRenderer",
    "build_templates",
    "estimate_metric_calls",
    "load_examples",
    "optimize_descriptions",
    "score_candidate",
    "score_document",
    "split",
]

"""Rasterising documents for the reflection model.

GEPA can only put images into a reflection prompt. Its only multimodal content
part is ``image_url`` (``gepa/image.py``), and it has no document/file part at
all -- verified against gepa 0.1.4. Worse, ``Image(path="x.pdf")`` does not
fail: ``_guess_media_type`` falls back to ``image/png`` for unknown extensions
and ``__post_init__`` never validates, so PDF bytes ship silently mislabelled
as a PNG data URI.

So documents are rasterised to PNG before they reach reflection, and the
media type is always passed **explicitly** downstream -- never inferred from a
file extension. Extraction is unaffected: the extractor receives the native
PDF, and only the reflection path rasterises.

Whole pages are rendered, never crops. Extraction returns values, not
coordinates, so there is no region to crop to -- and cropping would defeat the
purpose, which is to let a strong vision model do the localisation the weaker
extraction model could not.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from gepa_extract.documents import Document

__all__ = ["NullRenderer", "PageRenderer", "PdfiumRenderer", "RenderedPage", "StubRenderer"]


@dataclass(frozen=True, slots=True)
class RenderedPage:
    page_number: int  # 1-based, as a human would cite it
    path: Path
    media_type: str = "image/png"


class PageRenderer(Protocol):
    """Renders a document to page images for the reflection model."""

    def render(self, document: Document, *, max_pages: int | None = None) -> list[RenderedPage]: ...


class PdfiumRenderer:
    """Renders PDFs via pypdfium2. Install with ``pip install gepa-extract[render]``.

    Rendered pages are cached on disk by (document bytes, scale), so the same
    page is rasterised once per run rather than once per reflection.
    """

    def __init__(self, cache_dir: str | Path, *, scale: float = 2.0) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.scale = scale

    def render(self, document: Document, *, max_pages: int | None = None) -> list[RenderedPage]:
        try:
            import pypdfium2
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "PdfiumRenderer requires pypdfium2 and Pillow. Install with: pip install 'gepa-extract[render]'"
            ) from exc

        data = document.read_bytes()
        digest = hashlib.sha256(data).hexdigest()[:16]
        pdf = pypdfium2.PdfDocument(data)
        try:
            count = len(pdf) if max_pages is None else min(len(pdf), max_pages)
            pages: list[RenderedPage] = []
            for index in range(count):
                out_path = self.cache_dir / f"{digest}-s{self.scale:g}-p{index + 1}.png"
                if not out_path.exists():
                    bitmap = pdf[index].render(scale=self.scale)
                    bitmap.to_pil().save(out_path, format="PNG")
                pages.append(RenderedPage(page_number=index + 1, path=out_path))
            return pages
        finally:
            pdf.close()


class StubRenderer:
    """Returns fixed page images without touching a PDF. For tests and examples."""

    def __init__(self, pages: list[RenderedPage] | None = None) -> None:
        self._pages = pages or []

    def render(self, document: Document, *, max_pages: int | None = None) -> list[RenderedPage]:
        pages = self._pages
        return pages if max_pages is None else pages[:max_pages]


class NullRenderer:
    """Renders nothing -- reflection falls back to text-only evidence.

    Useful to measure what the visual channel is actually buying: run once with
    a real renderer and once with this, and compare.
    """

    def render(self, document: Document, *, max_pages: int | None = None) -> list[RenderedPage]:
        return []

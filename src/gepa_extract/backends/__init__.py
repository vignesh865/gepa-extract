"""Concrete extraction and reflection backends.

Backends are optional dependencies and import their SDKs lazily, so the core
package installs and tests without any provider library present.
"""

from gepa_extract.backends.gemini import GeminiExtractor, GeminiReflectionLM

__all__ = ["GeminiExtractor", "GeminiReflectionLM"]

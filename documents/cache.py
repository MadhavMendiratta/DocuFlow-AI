"""
Redis-backed caching utilities for document processing.

Generates a SHA-256 content hash for each uploaded file and uses Django's
cache framework (backed by Redis) to store / retrieve:

* **extracted text** — avoids re-running PDF/DOCX/TXT extraction
* **AI analysis results** — avoids duplicate Gemini API calls

Cache keys follow the pattern::

    doc:text:<sha256>        →  extracted text (str)
    doc:analysis:<sha256>    →  analysis dict (JSON-serialisable)

The default TTL is controlled by ``DOCUMENT_CACHE_TTL`` in settings
(defaults to 7 days).
"""

import hashlib
import json
import logging
from typing import Any, Dict, Optional

from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

# How long cached results survive (seconds).  Override in settings.py.
CACHE_TTL: int = getattr(settings, 'DOCUMENT_CACHE_TTL', 60 * 60 * 24 * 7)  # 7 days

# Key prefixes
_TEXT_PREFIX = 'doc:text:'
_ANALYSIS_PREFIX = 'doc:analysis:'


# ── Hashing ──────────────────────────────────────────────────────────────────


def compute_file_hash(file_field) -> str:
    """Return the SHA-256 hex digest of a Django ``FieldFile``'s content.

    The file is read in 8 KiB chunks so arbitrarily large files can be
    hashed without loading them entirely into memory.  The file position
    is reset to 0 afterwards so subsequent reads are unaffected.
    """
    hasher = hashlib.sha256()
    try:
        file_field.open('rb')
        for chunk in file_field.chunks(chunk_size=8192):
            hasher.update(chunk)
    finally:
        file_field.close()
    return hasher.hexdigest()


# ── Text-cache helpers ───────────────────────────────────────────────────────


def _text_key(file_hash: str) -> str:
    return f'{_TEXT_PREFIX}{file_hash}'


def get_cached_text(file_hash: str) -> Optional[str]:
    """Return cached extracted text for *file_hash*, or ``None``."""
    key = _text_key(file_hash)
    value = cache.get(key)
    if value is not None:
        logger.info(f"Cache HIT for extracted text ({key})")
    return value


def set_cached_text(file_hash: str, text: str) -> None:
    """Store extracted text in cache."""
    key = _text_key(file_hash)
    cache.set(key, text, timeout=CACHE_TTL)
    logger.debug(f"Cached extracted text ({key}, {len(text):,} chars, TTL={CACHE_TTL}s)")


# ── Analysis-cache helpers ───────────────────────────────────────────────────


def _analysis_key(file_hash: str) -> str:
    return f'{_ANALYSIS_PREFIX}{file_hash}'


def get_cached_analysis(file_hash: str) -> Optional[Dict[str, Any]]:
    """Return cached analysis dict for *file_hash*, or ``None``."""
    key = _analysis_key(file_hash)
    raw = cache.get(key)
    if raw is not None:
        logger.info(f"Cache HIT for analysis ({key})")
        return json.loads(raw) if isinstance(raw, str) else raw
    return None


def set_cached_analysis(file_hash: str, analysis_data: Dict[str, Any]) -> None:
    """Store analysis results in cache.

    *analysis_data* must be JSON-serialisable.
    """
    key = _analysis_key(file_hash)
    cache.set(key, json.dumps(analysis_data, default=str), timeout=CACHE_TTL)
    logger.debug(f"Cached analysis ({key}, TTL={CACHE_TTL}s)")


# ── Convenience: full result (text + analysis in one call) ───────────────────


def get_cached_result(file_hash: str) -> Optional[Dict[str, Any]]:
    """Return both cached text and analysis if **both** exist, else ``None``.

    Returns::

        {
            'extracted_text': str,
            'analysis': { ... },
        }
    """
    text = get_cached_text(file_hash)
    analysis = get_cached_analysis(file_hash)
    if text is not None and analysis is not None:
        return {'extracted_text': text, 'analysis': analysis}
    return None


def invalidate_cache(file_hash: str) -> None:
    """Remove all cached entries for a given file hash."""
    cache.delete(_text_key(file_hash))
    cache.delete(_analysis_key(file_hash))
    logger.info(f"Invalidated cache for hash {file_hash[:12]}…")
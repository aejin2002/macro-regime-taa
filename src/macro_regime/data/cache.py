"""Local disk cache for raw FRED API responses.

Caching is deliberately dumb and file-based: one JSON file per
(series_id, observation_start, observation_end, realtime_start, realtime_end)
combination. This keeps re-runs of `fetch` fast and avoids hammering the
FRED API during development, without introducing a database dependency.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

# Bumped whenever the on-disk payload SHAPE changes (not on every data
# refresh) -- lets a future reader detect "this file predates a schema
# change" distinctly from "this file is just old data". Purely additive
# metadata; existing readers that only look at `payload["records"]` (the
# original contract) are unaffected.
CACHE_SCHEMA_VERSION = "1"


def _cache_key(series_id: str, params: dict[str, Any]) -> str:
    normalized = json.dumps({"series_id": series_id, **params}, sort_keys=True, default=str)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{series_id}_{digest}"


class FileCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, series_id: str, params: dict[str, Any]) -> Path:
        return self.cache_dir / f"{_cache_key(series_id, params)}.json"

    def get(self, series_id: str, params: dict[str, Any]) -> dict[str, Any] | None:
        path = self._path(series_id, params)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
        return data

    def set(self, series_id: str, params: dict[str, Any], payload: dict[str, Any]) -> None:
        """Stores `payload` plus additive `_retrieved_at`/`_cache_version`
        metadata -- never overwrites a caller-supplied key of the same
        name (payload always wins), so this cannot change any existing
        reader's behavior."""
        path = self._path(series_id, params)
        enriched = {
            "_retrieved_at": pd.Timestamp.now().isoformat(),
            "_cache_version": CACHE_SCHEMA_VERSION,
            **payload,
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(enriched, fh)

    def cache_status(
        self, series_id: str, params: dict[str, Any], *, ttl_seconds: float | None = None
    ) -> dict[str, Any] | None:
        """Retrieved timestamp / cache version / TTL-based staleness for
        an existing cache entry, without touching or reinterpreting its
        `records` payload. Returns None if nothing is cached yet."""
        cached = self.get(series_id, params)
        if cached is None:
            return None
        retrieved_at_raw = cached.get("_retrieved_at")
        retrieved_at = pd.Timestamp(retrieved_at_raw) if retrieved_at_raw else None
        age_seconds = (
            (pd.Timestamp.now() - retrieved_at).total_seconds() if retrieved_at is not None else None
        )
        is_stale = bool(ttl_seconds is not None and age_seconds is not None and age_seconds > ttl_seconds)
        return {
            "retrieved_at": retrieved_at,
            "cache_version": cached.get("_cache_version"),
            "age_seconds": age_seconds,
            "ttl_seconds": ttl_seconds,
            "is_stale": is_stale,
        }

    def clear(self) -> None:
        for f in self.cache_dir.glob("*.json"):
            f.unlink()

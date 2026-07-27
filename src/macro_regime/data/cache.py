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
        path = self._path(series_id, params)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    def clear(self) -> None:
        for f in self.cache_dir.glob("*.json"):
            f.unlink()

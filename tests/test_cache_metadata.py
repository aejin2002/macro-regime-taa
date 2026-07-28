"""Tests for the cache metadata added during the v1.3 release-candidate
QA pass: retrieved timestamp, cache version, TTL-based staleness, and
`AssetPriceClient.get_cache_status`'s last-market-date/staleness report.
No network call is made -- these exercise `FileCache` directly against a
temp directory, and `AssetPriceClient` with `use_cache=False` fetch calls
monkeypatched out entirely (status is read straight from a pre-seeded
cache file).
"""

from __future__ import annotations

import time

from macro_regime.data.asset_prices import AssetPriceClient
from macro_regime.data.cache import CACHE_SCHEMA_VERSION, FileCache


def test_set_injects_retrieved_at_and_version(tmp_path):
    cache = FileCache(tmp_path)
    cache.set("SPY", {"start": "2020-01-01"}, {"records": [{"date": "2020-01-02", "close": 100.0}]})
    stored = cache.get("SPY", {"start": "2020-01-01"})
    assert "_retrieved_at" in stored
    assert stored["_cache_version"] == CACHE_SCHEMA_VERSION
    assert stored["records"] == [{"date": "2020-01-02", "close": 100.0}]  # payload untouched


def test_set_never_lets_metadata_override_a_same_named_payload_key(tmp_path):
    cache = FileCache(tmp_path)
    cache.set("SPY", {"start": "2020-01-01"}, {"_retrieved_at": "caller-supplied", "records": []})
    stored = cache.get("SPY", {"start": "2020-01-01"})
    assert stored["_retrieved_at"] == "caller-supplied"


def test_cache_status_none_when_nothing_cached(tmp_path):
    cache = FileCache(tmp_path)
    assert cache.cache_status("SPY", {"start": "2020-01-01"}) is None


def test_cache_status_not_stale_within_ttl(tmp_path):
    cache = FileCache(tmp_path)
    cache.set("SPY", {"start": "2020-01-01"}, {"records": []})
    status = cache.cache_status("SPY", {"start": "2020-01-01"}, ttl_seconds=3600)
    assert status["is_stale"] is False
    assert status["age_seconds"] is not None
    assert status["age_seconds"] < 5  # just written


def test_cache_status_stale_once_ttl_exceeded(tmp_path):
    cache = FileCache(tmp_path)
    cache.set("SPY", {"start": "2020-01-01"}, {"records": []})
    status = cache.cache_status("SPY", {"start": "2020-01-01"}, ttl_seconds=0.01)
    time.sleep(0.05)
    status = cache.cache_status("SPY", {"start": "2020-01-01"}, ttl_seconds=0.01)
    assert status["is_stale"] is True


def test_cache_status_no_ttl_never_stale(tmp_path):
    cache = FileCache(tmp_path)
    cache.set("SPY", {"start": "2020-01-01"}, {"records": []})
    status = cache.cache_status("SPY", {"start": "2020-01-01"}, ttl_seconds=None)
    assert status["is_stale"] is False


def test_asset_price_client_get_cache_status_reports_last_market_date(tmp_path):
    cache = FileCache(tmp_path)
    client = AssetPriceClient(cache=cache)
    cache.set(
        "SPY", {"start": "2020-01-01"},
        {"records": [{"date": "2026-07-20", "close": 100.0}, {"date": "2026-07-27", "close": 101.0}]},
    )
    status = client.get_cache_status("SPY", "2020-01-01")
    assert status["last_market_date"] == "2026-07-27"
    assert status["ticker"] == "SPY"


def test_asset_price_client_get_cache_status_none_for_uncached_ticker(tmp_path):
    cache = FileCache(tmp_path)
    client = AssetPriceClient(cache=cache)
    assert client.get_cache_status("AGG", "2020-01-01") is None


def test_asset_price_client_get_cache_status_different_start_is_independent(tmp_path):
    """Documents the known cache-fragmentation behavior (found during
    v1.3 QA): the SAME ticker with a DIFFERENT `start` is a completely
    separate cache entry with its own staleness."""
    cache = FileCache(tmp_path)
    client = AssetPriceClient(cache=cache)
    cache.set("SPY", {"start": "2000-01-01"}, {"records": [{"date": "2026-07-01", "close": 100.0}]})
    cache.set("SPY", {"start": "2009-01-01"}, {"records": [{"date": "2026-07-27", "close": 101.0}]})
    status_old = client.get_cache_status("SPY", "2000-01-01")
    status_new = client.get_cache_status("SPY", "2009-01-01")
    assert status_old["last_market_date"] != status_new["last_market_date"]

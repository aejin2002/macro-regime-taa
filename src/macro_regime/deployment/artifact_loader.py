"""Deployment-safe artifact resolution for the dashboard's parquet inputs.

Streamlit deployments check out this git repository, which never
includes `data/processed/*` (gitignored/untracked -- these are
pipeline-generated outputs, not source code). Without this module, a
fresh deployment has no way to obtain those artifacts and the dashboard
either fails at startup or silently shows "unavailable" for whatever
depends on them.

This module resolves WHERE to read each artifact from, independently
per asset:

1. The normal local path (e.g. `data/processed/production_v13_daily.parquet`)
   if it exists and validates -- the common case for local development,
   where `python -m macro_regime.cli update-all` has already been run.
2. Otherwise, downloads the EXACT asset attached to the GitHub Release
   for tag `v1.3.0` (never "latest", never a branch reference) into a
   writable runtime cache directory, verifies its SHA256 and schema, and
   returns that path instead.

This module NEVER runs `update-all`, NEVER runs a backtest, and NEVER
silently substitutes a different/partial version of an artifact if the
resolved file doesn't validate -- any such situation raises a typed
exception instead. The caller (Streamlit) decides how to surface that;
this module only ever resolves a path or fails loudly.

Two independent entry points, one per asset:
- `resolve_artifact_path()` -- `production_v13_daily.parquet`.
- `resolve_benchmarks_artifact_path()` -- `benchmarks_daily.parquet`
  (US 60/40, MALOX, SPY, AGG). Both share the same download/cache/atomic-
  write machinery and the same GitHub Release tag, but are validated
  against their own schema and resolved completely independently -- one
  being local-only or remote-only never affects the other.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from macro_regime.config import PROJECT_ROOT

GITHUB_OWNER = "aejin2002"
GITHUB_REPO = "macro-regime-taa"
RELEASE_TAG = "v1.3.0"
ASSET_NAME = "production_v13_daily.parquet"
EXPECTED_STRATEGY_VERSION = "v1_3"

# Pinned to the EXACT bytes uploaded to the v1.3.0 Release asset. A future
# release (v1.3.1, v1.4.0, ...) that needs a different artifact gets its
# own tag and its own updated constant here -- this is intentionally not
# looked up dynamically, so a compromised or mutated release asset can
# never be silently accepted.
EXPECTED_SHA256 = "2bd58aea781f0bc888c705dff7a3dd5a9db3d3a4e8f30ca8efe86a0ad478d29c"

# A representative subset of the full 63-column schema -- enough to catch
# a genuinely wrong or corrupt file without hard-coding every per-asset
# column (which would break the moment a new asset enters the universe).
REQUIRED_COLUMNS = (
    "date",
    "strategy_version",
    "daily_return_gross",
    "daily_return_net",
    "strategy_nav",
    "strategy_drawdown",
    "macro_regime",
    "growth_state",
    "inflation_state",
    "growth_state_observed",
    "inflation_state_observed",
    "bei_gate_raw",
    "bei_gate_tradable",
    "fast_crisis_raw",
    "fast_crisis_tradable",
    "fast_crisis_votes",
    "daily_turnover",
    "transaction_cost",
    "rebalance_event",
    "signal_event",
    "partial_month",
)

LOCAL_PATH = PROJECT_ROOT / "data" / "processed" / ASSET_NAME

BENCHMARKS_ASSET_NAME = "benchmarks_daily.parquet"
BENCHMARKS_LOCAL_PATH = PROJECT_ROOT / "data" / "processed" / BENCHMARKS_ASSET_NAME

# Pinned to the EXACT bytes uploaded to the v1.3.0 Release asset, same
# rationale as EXPECTED_SHA256 above.
BENCHMARKS_EXPECTED_SHA256 = "c0879e360662e214f0ebf8a68b29c237ebc128083f7bf3ca999efaa4695dd76c"

BENCHMARKS_REQUIRED_COLUMNS = ("date", "benchmark_id", "daily_return", "nav", "available")

# The UI-visible benchmarks (`macro_regime.benchmarks.list_ui_visible()`)
# that must be present for the dashboard's Comparison/Markets sections to
# work. `project_6040` (internal/research-only, `ui_visible=False`) is
# deliberately not required here.
BENCHMARKS_REQUIRED_IDS = ("us_60_40", "malox", "spy", "agg")


class ArtifactDownloadError(RuntimeError):
    """Network/HTTP failure fetching the release asset."""


class ArtifactValidationError(RuntimeError):
    """The resolved file exists but fails schema/version/checksum checks."""


@dataclass(frozen=True)
class ResolvedArtifact:
    path: Path
    source: str  # "local" | "github_release"
    tag: str | None
    sha256: str


def _runtime_cache_dir() -> Path:
    """A writable directory in every deployment environment this project
    targets (local dev, Streamlit Community Cloud, or any other
    read-only-repo-checkout host) -- never assumes the repo checkout
    itself is writable. `MACRO_REGIME_CACHE_DIR` lets a deployment pin
    this explicitly; otherwise falls back to the platform temp dir."""
    base = Path(os.environ.get("MACRO_REGIME_CACHE_DIR", tempfile.gettempdir()))
    cache_dir = base / "macro_regime_taa_artifact_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _release_asset_url(asset_name: str = ASSET_NAME) -> str:
    """Explicit, immutable URL for the exact asset attached to tag
    v1.3.0. GitHub's `.../releases/download/<tag>/<asset>` form is tied
    to the TAG, never to "latest" -- a future release can never silently
    change what this resolves to."""
    return f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/{asset_name}"


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_parquet(path: Path) -> None:
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        raise ArtifactValidationError(
            f"File at {path} is not a readable parquet file: {exc}"
        ) from exc

    if df.empty:
        raise ArtifactValidationError(f"Artifact at {path} has zero rows.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ArtifactValidationError(f"Artifact at {path} is missing required columns: {missing}")

    versions = set(df["strategy_version"].dropna().unique().tolist())
    if versions != {EXPECTED_STRATEGY_VERSION}:
        raise ArtifactValidationError(
            f"Artifact at {path} has strategy_version {sorted(versions)}, expected exactly "
            f"{{'{EXPECTED_STRATEGY_VERSION}'}} -- refusing to silently substitute a different version."
        )


def _validate_benchmarks_parquet(path: Path) -> None:
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        raise ArtifactValidationError(
            f"File at {path} is not a readable parquet file: {exc}"
        ) from exc

    if df.empty:
        raise ArtifactValidationError(f"Benchmarks artifact at {path} has zero rows.")

    missing = [c for c in BENCHMARKS_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ArtifactValidationError(f"Benchmarks artifact at {path} is missing required columns: {missing}")

    ids = set(df["benchmark_id"].dropna().unique().tolist())
    missing_ids = [b for b in BENCHMARKS_REQUIRED_IDS if b not in ids]
    if missing_ids:
        raise ArtifactValidationError(
            f"Benchmarks artifact at {path} is missing required benchmark_id(s): {missing_ids} "
            "-- refusing to silently substitute a partial benchmark set."
        )


def _download_and_validate(
    *,
    asset_name: str,
    validate_fn,
    expected_sha256: str | None,
    timeout_seconds: float,
) -> Path:
    cache_dir = _runtime_cache_dir()
    final_path = cache_dir / asset_name
    marker_path = cache_dir / f"{asset_name}.{RELEASE_TAG}.ok"

    # Avoid re-downloading on every Streamlit rerun: reuse a prior
    # successful download for THIS EXACT tag if it's still present and
    # still validates.
    if final_path.exists() and marker_path.exists():
        try:
            validate_fn(final_path)
            return final_path
        except ArtifactValidationError:
            pass  # cached copy is bad -- fall through and re-download

    url = _release_asset_url(asset_name)
    try:
        response = requests.get(url, stream=True, timeout=timeout_seconds, allow_redirects=True)
    except requests.RequestException as exc:
        raise ArtifactDownloadError(f"Failed to reach {url}: {exc}") from exc

    if response.status_code != 200:
        raise ArtifactDownloadError(
            f"GitHub Release download returned HTTP {response.status_code} for {url} "
            f"(tag={RELEASE_TAG}, asset={asset_name})."
        )

    # Atomic: write to a temp file in the SAME directory (so the final
    # replace is a same-filesystem rename, not a copy-then-delete), then
    # rename into place -- a concurrent reader can never observe a
    # partially-written file at `final_path`.
    fd, tmp_path_str = tempfile.mkstemp(dir=cache_dir, prefix=f".{asset_name}.", suffix=".part")
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)

        actual_sha256 = _sha256_of(tmp_path)
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ArtifactValidationError(
                f"SHA256 mismatch for downloaded {asset_name}: "
                f"expected {expected_sha256}, got {actual_sha256}."
            )
        validate_fn(tmp_path)  # validate BEFORE the atomic rename -- final_path never sees a bad file
        os.replace(tmp_path, final_path)
        marker_path.write_text(actual_sha256)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return final_path


def resolve_artifact_path(
    *, expected_sha256: str | None = EXPECTED_SHA256, timeout_seconds: float = 30.0
) -> ResolvedArtifact:
    """The single entry point Streamlit's data loader calls. Never runs
    `update-all`, never runs a backtest -- only ever reads an existing
    local file or downloads an already-built one from the pinned
    GitHub Release. Raises `ArtifactDownloadError`/`ArtifactValidationError`
    on any failure rather than returning a guessed or partial path."""
    if LOCAL_PATH.exists():
        _validate_parquet(LOCAL_PATH)  # raises if bad -- no silent remote fallback for a corrupt LOCAL file
        return ResolvedArtifact(path=LOCAL_PATH, source="local", tag=None, sha256=_sha256_of(LOCAL_PATH))

    downloaded_path = _download_and_validate(
        asset_name=ASSET_NAME,
        validate_fn=_validate_parquet,
        expected_sha256=expected_sha256,
        timeout_seconds=timeout_seconds,
    )
    return ResolvedArtifact(
        path=downloaded_path, source="github_release", tag=RELEASE_TAG, sha256=_sha256_of(downloaded_path)
    )


def resolve_benchmarks_artifact_path(
    *, expected_sha256: str | None = BENCHMARKS_EXPECTED_SHA256, timeout_seconds: float = 30.0
) -> ResolvedArtifact:
    """Same resolution contract as `resolve_artifact_path`, independently,
    for `benchmarks_daily.parquet` (US 60/40, MALOX, SPY, AGG). One
    artifact being local-only or remote-only never affects the other."""
    if BENCHMARKS_LOCAL_PATH.exists():
        _validate_benchmarks_parquet(BENCHMARKS_LOCAL_PATH)
        return ResolvedArtifact(
            path=BENCHMARKS_LOCAL_PATH, source="local", tag=None, sha256=_sha256_of(BENCHMARKS_LOCAL_PATH)
        )

    downloaded_path = _download_and_validate(
        asset_name=BENCHMARKS_ASSET_NAME,
        validate_fn=_validate_benchmarks_parquet,
        expected_sha256=expected_sha256,
        timeout_seconds=timeout_seconds,
    )
    return ResolvedArtifact(
        path=downloaded_path, source="github_release", tag=RELEASE_TAG, sha256=_sha256_of(downloaded_path)
    )

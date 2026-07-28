"""Tests for the deployment-safe artifact loader
(`macro_regime.deployment.artifact_loader`). No live network call is
made -- HTTP is mocked via `responses` (already a dev dependency,
`pyproject.toml`'s `[project.optional-dependencies].dev`). All tests
point `LOCAL_PATH`/the runtime cache dir at pytest's own `tmp_path`
fixture via monkeypatch, never touching the real project's
`data/processed/` or the OS temp dir.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest
import responses

from macro_regime.deployment import artifact_loader as al


def _write_valid_parquet(path: Path, *, strategy_version: str = "v1_3", n_rows: int = 3) -> bytes:
    df = pd.DataFrame(
        {col: [0] * n_rows for col in al.REQUIRED_COLUMNS if col not in ("date", "strategy_version")}
    )
    df["date"] = pd.date_range("2026-01-01", periods=n_rows)
    df["strategy_version"] = strategy_version
    df.to_parquet(path, index=False)
    return path.read_bytes()


def _write_valid_benchmarks_parquet(
    path: Path, *, benchmark_ids: tuple[str, ...] = al.BENCHMARKS_REQUIRED_IDS
) -> bytes:
    rows = []
    for bid in benchmark_ids:
        for d in pd.date_range("2026-01-01", periods=3):
            rows.append({"date": d, "benchmark_id": bid, "daily_return": 0.0, "nav": 1.0, "available": True})
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path.read_bytes()


@pytest.fixture(autouse=True)
def _isolated_paths(tmp_path, monkeypatch):
    """Every test gets its own local-path location and its own cache
    dir -- never the real project paths."""
    local_path = tmp_path / "local" / al.ASSET_NAME
    bench_local_path = tmp_path / "local" / al.BENCHMARKS_ASSET_NAME
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(al, "LOCAL_PATH", local_path)
    monkeypatch.setattr(al, "BENCHMARKS_LOCAL_PATH", bench_local_path)
    monkeypatch.setattr(al, "_runtime_cache_dir", lambda: cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {"local_path": local_path, "bench_local_path": bench_local_path, "cache_dir": cache_dir}


# -- local artifact exists ---------------------------------------------------


def test_local_artifact_used_directly_no_network_call(_isolated_paths):
    local_path = _isolated_paths["local_path"]
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _write_valid_parquet(local_path)

    # No `responses.activate` here at all -- if the code tried to make a
    # real HTTP call, `requests` would attempt a live connection and this
    # test would hang/fail rather than silently succeed.
    resolved = al.resolve_artifact_path()
    assert resolved.source == "local"
    assert resolved.path == local_path


def test_local_artifact_corrupt_does_not_silently_fall_back_to_remote(_isolated_paths):
    local_path = _isolated_paths["local_path"]
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(b"not a parquet file")

    with pytest.raises(al.ArtifactValidationError):
        al.resolve_artifact_path()


# -- local missing, remote download succeeds ---------------------------------


@responses.activate
def test_local_missing_remote_download_succeeds(_isolated_paths):
    content = _write_valid_parquet(Path(_isolated_paths["cache_dir"]) / "source_for_upload.parquet")
    responses.add(responses.GET, al._release_asset_url(), body=content, status=200)

    resolved = al.resolve_artifact_path(expected_sha256=None)
    assert resolved.source == "github_release"
    assert resolved.tag == al.RELEASE_TAG
    assert resolved.path.exists()
    assert resolved.path.read_bytes() == content


@responses.activate
def test_downloaded_artifact_is_readable_and_correct(_isolated_paths):
    content = _write_valid_parquet(Path(_isolated_paths["cache_dir"]) / "source.parquet", n_rows=5)
    responses.add(responses.GET, al._release_asset_url(), body=content, status=200)

    resolved = al.resolve_artifact_path(expected_sha256=None)
    df = pd.read_parquet(resolved.path)
    assert len(df) == 5
    assert set(df["strategy_version"].unique()) == {"v1_3"}


# -- HTTP failure -------------------------------------------------------------


@responses.activate
def test_http_404_raises_download_error(_isolated_paths):
    responses.add(responses.GET, al._release_asset_url(), status=404)
    with pytest.raises(al.ArtifactDownloadError, match="404"):
        al.resolve_artifact_path(expected_sha256=None)


@responses.activate
def test_http_500_raises_download_error(_isolated_paths):
    responses.add(responses.GET, al._release_asset_url(), status=500)
    with pytest.raises(al.ArtifactDownloadError, match="500"):
        al.resolve_artifact_path(expected_sha256=None)


def test_connection_error_raises_download_error(_isolated_paths, monkeypatch):
    import requests

    def _raise(*args, **kwargs):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(al.requests, "get", _raise)
    with pytest.raises(al.ArtifactDownloadError, match="no network"):
        al.resolve_artifact_path(expected_sha256=None)


# -- checksum mismatch --------------------------------------------------------


@responses.activate
def test_checksum_mismatch_raises_validation_error(_isolated_paths):
    content = _write_valid_parquet(Path(_isolated_paths["cache_dir"]) / "source.parquet")
    responses.add(responses.GET, al._release_asset_url(), body=content, status=200)

    with pytest.raises(al.ArtifactValidationError, match="SHA256 mismatch"):
        al.resolve_artifact_path(expected_sha256="0" * 64)


@responses.activate
def test_checksum_mismatch_leaves_no_final_file(_isolated_paths):
    content = _write_valid_parquet(Path(_isolated_paths["cache_dir"]) / "source.parquet")
    responses.add(responses.GET, al._release_asset_url(), body=content, status=200)

    with pytest.raises(al.ArtifactValidationError):
        al.resolve_artifact_path(expected_sha256="0" * 64)

    final_path = _isolated_paths["cache_dir"] / al.ASSET_NAME
    assert not final_path.exists()


# -- corrupt parquet -----------------------------------------------------------


@responses.activate
def test_corrupt_downloaded_file_raises_validation_error(_isolated_paths):
    responses.add(responses.GET, al._release_asset_url(), body=b"this is not parquet", status=200)
    with pytest.raises(al.ArtifactValidationError, match="not a readable parquet"):
        al.resolve_artifact_path(expected_sha256=None)


# -- missing required columns --------------------------------------------------


@responses.activate
def test_missing_required_columns_raises_validation_error(_isolated_paths):
    incomplete = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=2), "strategy_version": "v1_3"})
    buf = io.BytesIO()
    incomplete.to_parquet(buf, index=False)
    responses.add(responses.GET, al._release_asset_url(), body=buf.getvalue(), status=200)

    with pytest.raises(al.ArtifactValidationError, match="missing required columns"):
        al.resolve_artifact_path(expected_sha256=None)


# -- wrong strategy version -----------------------------------------------------


@responses.activate
def test_wrong_strategy_version_raises_validation_error_not_silent_fallback(_isolated_paths):
    content = _write_valid_parquet(
        Path(_isolated_paths["cache_dir"]) / "source.parquet", strategy_version="v1_2"
    )
    responses.add(responses.GET, al._release_asset_url(), body=content, status=200)

    with pytest.raises(al.ArtifactValidationError, match="strategy_version"):
        al.resolve_artifact_path(expected_sha256=None)


def test_local_wrong_version_also_raises_not_silent_fallback(_isolated_paths):
    local_path = _isolated_paths["local_path"]
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _write_valid_parquet(local_path, strategy_version="v1_2")

    with pytest.raises(al.ArtifactValidationError, match="strategy_version"):
        al.resolve_artifact_path()


# -- atomic download behavior ----------------------------------------------------


@responses.activate
def test_successful_download_leaves_no_temp_part_files(_isolated_paths):
    content = _write_valid_parquet(Path(_isolated_paths["cache_dir"]) / "source.parquet")
    responses.add(responses.GET, al._release_asset_url(), body=content, status=200)

    al.resolve_artifact_path(expected_sha256=None)
    leftover = list(_isolated_paths["cache_dir"].glob("*.part"))
    assert leftover == []


@responses.activate
def test_failed_validation_leaves_no_temp_part_files(_isolated_paths):
    responses.add(responses.GET, al._release_asset_url(), body=b"garbage", status=200)
    with pytest.raises(al.ArtifactValidationError):
        al.resolve_artifact_path(expected_sha256=None)
    leftover = list(_isolated_paths["cache_dir"].glob("*.part"))
    assert leftover == []


@responses.activate
def test_repeated_resolve_does_not_redownload(_isolated_paths):
    content = _write_valid_parquet(Path(_isolated_paths["cache_dir"]) / "source.parquet")
    responses.add(responses.GET, al._release_asset_url(), body=content, status=200)
    responses.add(responses.GET, al._release_asset_url(), body=content, status=200)

    al.resolve_artifact_path(expected_sha256=None)
    al.resolve_artifact_path(expected_sha256=None)  # second call must reuse the cached copy
    assert len(responses.calls) == 1


# -- Streamlit loader integration --------------------------------------------------


def test_streamlit_loader_uses_resolved_path(monkeypatch, tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
    from ui import data_loader

    fake_path = tmp_path / "resolved.parquet"
    _write_valid_parquet(fake_path)
    fake_resolved = al.ResolvedArtifact(
        path=fake_path, source="github_release", tag="v1.3.0", sha256="abc123"
    )

    data_loader._resolved_v1_3_artifact.clear()
    monkeypatch.setattr(data_loader, "_resolved_v1_3_artifact", lambda: fake_resolved)

    info = data_loader.v1_3_artifact_source_info()
    assert info["source"] == "github_release"
    assert info["tag"] == "v1.3.0"
    assert info["path"] == str(fake_path)

    df = data_loader.load_v1_3_daily()
    assert df is not None
    assert len(df) == 3


# =============================================================================
# benchmarks_daily.parquet -- same resolution contract, independent asset
# =============================================================================


def test_benchmarks_local_artifact_used_directly_no_network_call(_isolated_paths):
    bench_local_path = _isolated_paths["bench_local_path"]
    bench_local_path.parent.mkdir(parents=True, exist_ok=True)
    _write_valid_benchmarks_parquet(bench_local_path)

    resolved = al.resolve_benchmarks_artifact_path()
    assert resolved.source == "local"
    assert resolved.path == bench_local_path


def test_benchmarks_local_artifact_corrupt_does_not_silently_fall_back_to_remote(_isolated_paths):
    bench_local_path = _isolated_paths["bench_local_path"]
    bench_local_path.parent.mkdir(parents=True, exist_ok=True)
    bench_local_path.write_bytes(b"not a parquet file")

    with pytest.raises(al.ArtifactValidationError):
        al.resolve_benchmarks_artifact_path()


@responses.activate
def test_benchmarks_local_missing_remote_download_succeeds(_isolated_paths):
    content = _write_valid_benchmarks_parquet(Path(_isolated_paths["cache_dir"]) / "source.parquet")
    responses.add(responses.GET, al._release_asset_url(al.BENCHMARKS_ASSET_NAME), body=content, status=200)

    resolved = al.resolve_benchmarks_artifact_path(expected_sha256=None)
    assert resolved.source == "github_release"
    assert resolved.tag == al.RELEASE_TAG
    assert resolved.path.exists()
    assert resolved.path.read_bytes() == content


@responses.activate
def test_benchmarks_http_404_raises_download_error(_isolated_paths):
    responses.add(responses.GET, al._release_asset_url(al.BENCHMARKS_ASSET_NAME), status=404)
    with pytest.raises(al.ArtifactDownloadError, match="404"):
        al.resolve_benchmarks_artifact_path(expected_sha256=None)


@responses.activate
def test_benchmarks_checksum_mismatch_raises_validation_error(_isolated_paths):
    content = _write_valid_benchmarks_parquet(Path(_isolated_paths["cache_dir"]) / "source.parquet")
    responses.add(responses.GET, al._release_asset_url(al.BENCHMARKS_ASSET_NAME), body=content, status=200)

    with pytest.raises(al.ArtifactValidationError, match="SHA256 mismatch"):
        al.resolve_benchmarks_artifact_path(expected_sha256="0" * 64)


@responses.activate
def test_benchmarks_missing_required_columns_raises_validation_error(_isolated_paths):
    incomplete = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=2), "benchmark_id": "us_60_40"})
    buf = io.BytesIO()
    incomplete.to_parquet(buf, index=False)
    responses.add(
        responses.GET, al._release_asset_url(al.BENCHMARKS_ASSET_NAME), body=buf.getvalue(), status=200
    )

    with pytest.raises(al.ArtifactValidationError, match="missing required columns"):
        al.resolve_benchmarks_artifact_path(expected_sha256=None)


@responses.activate
def test_benchmarks_missing_required_id_raises_validation_error_not_silent_fallback(_isolated_paths):
    content = _write_valid_benchmarks_parquet(
        Path(_isolated_paths["cache_dir"]) / "source.parquet", benchmark_ids=("us_60_40", "spy")
    )
    responses.add(responses.GET, al._release_asset_url(al.BENCHMARKS_ASSET_NAME), body=content, status=200)

    with pytest.raises(al.ArtifactValidationError, match="missing required benchmark_id"):
        al.resolve_benchmarks_artifact_path(expected_sha256=None)


def test_benchmarks_local_missing_ids_also_raises_not_silent_fallback(_isolated_paths):
    bench_local_path = _isolated_paths["bench_local_path"]
    bench_local_path.parent.mkdir(parents=True, exist_ok=True)
    _write_valid_benchmarks_parquet(bench_local_path, benchmark_ids=("us_60_40",))

    with pytest.raises(al.ArtifactValidationError, match="missing required benchmark_id"):
        al.resolve_benchmarks_artifact_path()


@responses.activate
def test_benchmarks_successful_download_leaves_no_temp_part_files(_isolated_paths):
    content = _write_valid_benchmarks_parquet(Path(_isolated_paths["cache_dir"]) / "source.parquet")
    responses.add(responses.GET, al._release_asset_url(al.BENCHMARKS_ASSET_NAME), body=content, status=200)

    al.resolve_benchmarks_artifact_path(expected_sha256=None)
    leftover = list(_isolated_paths["cache_dir"].glob("*.part"))
    assert leftover == []


@responses.activate
def test_benchmarks_repeated_resolve_does_not_redownload(_isolated_paths):
    content = _write_valid_benchmarks_parquet(Path(_isolated_paths["cache_dir"]) / "source.parquet")
    responses.add(responses.GET, al._release_asset_url(al.BENCHMARKS_ASSET_NAME), body=content, status=200)
    responses.add(responses.GET, al._release_asset_url(al.BENCHMARKS_ASSET_NAME), body=content, status=200)

    al.resolve_benchmarks_artifact_path(expected_sha256=None)
    al.resolve_benchmarks_artifact_path(expected_sha256=None)
    assert len(responses.calls) == 1


def test_v13_and_benchmarks_resolution_are_independent(_isolated_paths):
    """v1.3 resolving from local while benchmarks is entirely missing (and
    vice versa) must not affect each other -- confirmed by resolving v1.3
    successfully while benchmarks has no local file and no HTTP mock."""
    local_path = _isolated_paths["local_path"]
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _write_valid_parquet(local_path)

    resolved = al.resolve_artifact_path()
    assert resolved.source == "local"
    assert not _isolated_paths["bench_local_path"].exists()


def test_streamlit_loader_benchmarks_uses_resolved_path(monkeypatch, tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
    from ui import data_loader

    fake_path = tmp_path / "resolved_benchmarks.parquet"
    _write_valid_benchmarks_parquet(fake_path)
    fake_resolved = al.ResolvedArtifact(
        path=fake_path, source="github_release", tag="v1.3.0", sha256="def456"
    )

    data_loader._resolved_benchmarks_artifact.clear()
    monkeypatch.setattr(data_loader, "_resolved_benchmarks_artifact", lambda: fake_resolved)

    info = data_loader.benchmarks_artifact_source_info()
    assert info["source"] == "github_release"
    assert info["tag"] == "v1.3.0"
    assert info["path"] == str(fake_path)

    df = data_loader.load_benchmarks_daily()
    assert df is not None
    assert set(df["benchmark_id"].unique()) == set(al.BENCHMARKS_REQUIRED_IDS)

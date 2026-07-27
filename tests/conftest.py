import pytest

from macro_regime.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(fred_api_key="test_api_key", cache_dir=tmp_path / "cache")

import json

import pytest
import responses

from macro_regime.config import MissingApiKeyError, Settings
from macro_regime.data.fred_client import FredClient

OBS_URL = "https://api.stlouisfed.org/fred/series/observations"


def test_missing_api_key_raises(tmp_path):
    settings = Settings(fred_api_key=None, cache_dir=tmp_path)
    with pytest.raises(MissingApiKeyError):
        FredClient(settings=settings)


@responses.activate
def test_get_series_parses_observations(settings):
    responses.add(
        responses.GET,
        OBS_URL,
        json={
            "observations": [
                {
                    "date": "2020-01-01",
                    "value": "100.5",
                    "realtime_start": "2020-02-01",
                    "realtime_end": "9999-12-31",
                },
                {
                    "date": "2020-02-01",
                    "value": "101.0",
                    "realtime_start": "2020-03-01",
                    "realtime_end": "9999-12-31",
                },
            ]
        },
        status=200,
    )
    client = FredClient(settings=settings, use_cache=False)
    df = client.get_series("INDPRO")

    assert list(df.columns) == ["date", "value", "realtime_start", "realtime_end", "series_id"]
    assert df.shape[0] == 2
    assert df["value"].iloc[0] == pytest.approx(100.5)
    assert (df["series_id"] == "INDPRO").all()


@responses.activate
def test_missing_value_dot_becomes_nan(settings):
    responses.add(
        responses.GET,
        OBS_URL,
        json={
            "observations": [
                {"date": "2020-01-01", "value": ".", "realtime_start": "x", "realtime_end": "y"}
            ]
        },
        status=200,
    )
    client = FredClient(settings=settings, use_cache=False)
    df = client.get_series("ICSA")
    assert df["value"].isna().all()


@responses.activate
def test_cache_avoids_second_network_call(settings):
    call_count = {"n": 0}

    def callback(request):
        call_count["n"] += 1
        payload = {
            "observations": [
                {"date": "2020-01-01", "value": "1.0", "realtime_start": "x", "realtime_end": "y"}
            ]
        }
        return 200, {}, json.dumps(payload)

    responses.add_callback(
        responses.GET, OBS_URL, callback=callback, content_type="application/json"
    )

    client = FredClient(settings=settings, use_cache=True)
    client.get_series("INDPRO")
    client.get_series("INDPRO")

    assert call_count["n"] == 1

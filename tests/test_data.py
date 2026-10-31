"""Recorded official API response plus deterministic transport/cache edge cases."""

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from src.data import DataUnavailable, parse_observations
from src.data import bank_of_canada as client
from src.data.validation import SERIES


@pytest.fixture
def payload():
    path = Path(__file__).parent / "fixtures" / "official_valet_observations.json"
    return json.loads(path.read_text())


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setattr(client, "_now", lambda: datetime(2026, 10, 31, 12, tzinfo=UTC))


def request_data(tmp_path, **kwargs):
    return client.load_data("2026-10-26", "2026-10-30", cache_dir=tmp_path, **kwargs)


def test_official_response_preserves_percent_and_sorts(payload):
    payload["observations"].reverse()
    frame, warnings = parse_observations(payload)
    assert list(frame.columns) == ["2Y", "3Y", "5Y", "7Y", "10Y"]
    assert frame.index.name == "date"
    assert frame.index.is_monotonic_increasing
    assert frame.loc["2026-10-26", "2Y"] == 3.37
    assert frame.loc["2026-10-29", "10Y"] == 3.84
    assert not warnings


def test_missing_invalid_nonfinite_and_no_forward_fill(payload):
    payload["observations"][1].pop(SERIES["2Y"])
    payload["observations"][2][SERIES["5Y"]]["v"] = "bad"
    payload["observations"][3][SERIES["10Y"]]["v"] = "Infinity"
    frame, warnings = parse_observations(payload)
    assert pd.isna(frame.loc["2026-10-27", "2Y"])
    assert pd.isna(frame.loc["2026-10-28", "5Y"])
    assert pd.isna(frame.loc["2026-10-29", "10Y"])
    assert any("invalid" in warning for warning in warnings)
    assert len(frame) == 4


def test_blank_calendar_days_removed_actual_weekend_retained(payload):
    payload["observations"].append({"d": "2026-10-31"})
    payload["observations"].append({"d": "2026-11-01", SERIES["2Y"]: {"v": "3.1"}})
    frame, warnings = parse_observations(payload)
    assert pd.Timestamp("2026-10-31") not in frame.index
    assert frame.loc["2026-11-01", "2Y"] == 3.1
    assert any("Omitted 1" in warning for warning in warnings)


@pytest.mark.parametrize(
    "bad_date", [None, "2026-02-30", "2026-10-26T00:00:00", "not-date", "0000-01-01"]
)
def test_invalid_dates_rejected(payload, bad_date):
    payload["observations"][0]["d"] = bad_date
    with pytest.raises(DataUnavailable, match="Invalid observation date"):
        parse_observations(payload)


def test_duplicate_dates_rejected_even_when_values_identical(payload):
    payload["observations"].append(copy.deepcopy(payload["observations"][0]))
    with pytest.raises(DataUnavailable, match="Duplicate"):
        parse_observations(payload)


@pytest.mark.parametrize(
    "change", ["missing_series", "wrong_tenor", "wrong_unit", "wrong_frequency"]
)
def test_metadata_verified(payload, change):
    details = payload["seriesDetail"]
    detail = details[SERIES["2Y"]]
    if change == "missing_series":
        details.pop(SERIES["2Y"])
    elif change == "wrong_tenor":
        detail["label"] = "Benchmark bond yield, 10-year"
    elif change == "wrong_unit":
        detail["metadata"]["unit"] = "Basis points"
    else:
        detail["metadata"]["frequency"] = "Monthly"
    with pytest.raises(DataUnavailable):
        parse_observations(payload)


def test_no_valid_observations_rejected(payload):
    payload["observations"] = [{"d": "2026-10-26"}]
    with pytest.raises(DataUnavailable, match="No numeric"):
        parse_observations(payload)


def test_live_then_fresh_cache_and_refresh(payload, tmp_path, monkeypatch):
    fetch = Mock(return_value=payload)
    monkeypatch.setattr(client, "_request", fetch)
    live = request_data(tmp_path)
    cached = request_data(tmp_path)
    assert live.source == "live"
    assert cached.source == "cache"
    assert fetch.call_count == 1
    assert cached.latest_date == pd.Timestamp("2026-10-29")
    assert cached.metadata["units"] == "percent"
    assert cached.metadata["series"] == SERIES
    pd.testing.assert_frame_equal(live.frame, cached.frame)
    assert request_data(tmp_path, refresh=True).source == "live"
    assert fetch.call_count == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_stale_cache_is_labelled_after_network_failure(payload, tmp_path, monkeypatch):
    monkeypatch.setattr(client, "_request", Mock(return_value=payload))
    first = request_data(tmp_path)
    monkeypatch.setattr(client, "_now", lambda: datetime(2026, 11, 1, 12, tzinfo=UTC))
    monkeypatch.setattr(client, "_request", Mock(side_effect=DataUnavailable("network timeout")))
    stale = request_data(tmp_path)
    assert stale.source == "stale cache"
    assert stale.fetched_at == first.fetched_at
    assert any("Live refresh failed" in warning for warning in stale.warnings)


def test_unavailable_with_no_cache_is_clear(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "_request", Mock(side_effect=DataUnavailable("offline")))
    with pytest.raises(DataUnavailable, match="No usable official data or validated cache"):
        request_data(tmp_path)


@pytest.mark.parametrize("corruption", ["json", "checksum", "coverage", "metadata", "timestamp"])
def test_corrupt_cache_never_used(payload, tmp_path, monkeypatch, corruption):
    monkeypatch.setattr(client, "_request", Mock(return_value=payload))
    request_data(tmp_path)
    path = next(tmp_path.glob("*.json"))
    envelope = json.loads(path.read_text())
    if corruption == "json":
        path.write_text("not json")
    else:
        if corruption == "checksum":
            envelope["payload"]["observations"][0][SERIES["2Y"]]["v"] = "999"
        elif corruption == "coverage":
            envelope["request"]["start_date"] = "2026-10-28"
        elif corruption == "metadata":
            envelope["payload"]["seriesDetail"].pop(SERIES["7Y"])
            envelope["sha256"] = client._digest(envelope["payload"])
        else:
            envelope["fetched_at"] = "2027-01-01T00:00:00+00:00"
        path.write_text(json.dumps(envelope))
    monkeypatch.setattr(client, "_request", Mock(side_effect=DataUnavailable("offline")))
    with pytest.raises(DataUnavailable, match="Ignored unusable local cache"):
        request_data(tmp_path)


def test_query_ranges_do_not_share_cache(payload, tmp_path, monkeypatch):
    monkeypatch.setattr(client, "_request", Mock(return_value=payload))
    request_data(tmp_path)
    monkeypatch.setattr(client, "_request", Mock(side_effect=DataUnavailable("offline")))
    with pytest.raises(DataUnavailable):
        client.load_data("2000-01-01", "2026-10-30", cache_dir=tmp_path)


def test_out_of_range_response_rejected(payload, tmp_path, monkeypatch):
    monkeypatch.setattr(client, "_request", Mock(return_value=payload))
    with pytest.raises(DataUnavailable, match="outside the requested"):
        client.load_data("2026-10-28", "2026-10-30", cache_dir=tmp_path)
    assert not list(tmp_path.glob("*.json"))


def test_bad_live_response_preserves_validated_cache(payload, tmp_path, monkeypatch):
    monkeypatch.setattr(client, "_request", Mock(return_value=payload))
    first = request_data(tmp_path)
    payload["seriesDetail"].pop(SERIES["5Y"])
    result = request_data(tmp_path, refresh=True)
    assert result.source == "stale cache"
    pd.testing.assert_frame_equal(result.frame, first.frame)


def test_cache_write_error_does_not_discard_good_live_data(payload, tmp_path, monkeypatch):
    monkeypatch.setattr(client, "_request", Mock(return_value=payload))
    monkeypatch.setattr(client, "_save_cache", Mock(side_effect=OSError("read only")))
    result = request_data(tmp_path)
    assert result.source == "live"
    assert any("could not be saved" in warning for warning in result.warnings)


def test_bounded_transient_retries_and_timeout(monkeypatch):
    get = Mock(side_effect=requests.Timeout("timed out"))
    sleep = Mock()
    monkeypatch.setattr(client.requests, "get", get)
    monkeypatch.setattr(client.time, "sleep", sleep)
    with pytest.raises(DataUnavailable, match="bounded retries"):
        client._request({"start_date": "2026-10-26"})
    assert get.call_count == 3
    assert sleep.call_count == 2
    assert get.call_args.kwargs["timeout"] == (5, 30)


def test_permanent_http_failure_not_retried(monkeypatch):
    response = Mock(status_code=404)
    response.raise_for_status.side_effect = requests.HTTPError(response=response)
    get = Mock(return_value=response)
    monkeypatch.setattr(client.requests, "get", get)
    with pytest.raises(DataUnavailable, match="HTTP 404"):
        client._request({})
    assert get.call_count == 1


def test_cache_expiry_at_six_hours(payload, tmp_path, monkeypatch):
    fetch = Mock(return_value=payload)
    monkeypatch.setattr(client, "_request", fetch)
    request_data(tmp_path)
    fixed = datetime(2026, 10, 31, 12, tzinfo=UTC)
    monkeypatch.setattr(client, "_now", lambda: fixed + timedelta(hours=6, seconds=1))
    assert request_data(tmp_path).source == "live"
    assert fetch.call_count == 2


@pytest.mark.parametrize(
    "values",
    [
        ["bad", "bad", "bad", "bad", "bad"],
        ["NaN", "Infinity", "-Infinity", "bad", True],
        ["bad", None, None, "", ""],
    ],
)
def test_corrupt_all_missing_observation_rejected_not_calendar_dropped(payload, values):
    corrupt = {"d": "2026-10-27"}
    for identifier, value in zip(SERIES.values(), values, strict=True):
        corrupt[identifier] = {"v": value}
    payload["observations"][1] = corrupt
    with pytest.raises(DataUnavailable, match="refusing to remove an invalid observation"):
        parse_observations(payload)


def test_malformed_all_missing_observation_rejected(payload):
    payload["observations"][1] = {"d": "2026-10-27", SERIES["2Y"]: "malformed cell"}
    with pytest.raises(DataUnavailable, match="corrupt supplied values"):
        parse_observations(payload)


def test_genuine_all_blank_publication_row_can_be_omitted(payload):
    payload["observations"][1] = {
        "d": "2026-10-27",
        **{identifier: {"v": ""} for identifier in SERIES.values()},
    }
    frame, warnings = parse_observations(payload)
    assert len(frame) == 3
    assert pd.Timestamp("2026-10-27") not in frame.index
    assert any("Omitted 1" in warning for warning in warnings)


def test_query_end_date_is_part_of_cache_key(payload, tmp_path, monkeypatch):
    monkeypatch.setattr(client, "_request", Mock(return_value=payload))
    request_data(tmp_path)
    monkeypatch.setattr(client, "_request", Mock(side_effect=DataUnavailable("offline")))
    with pytest.raises(DataUnavailable):
        client.load_data("2026-10-26", "2026-10-29", cache_dir=tmp_path)


def test_corrupted_calendar_row_never_overwrites_good_cache(payload, tmp_path, monkeypatch):
    monkeypatch.setattr(client, "_request", Mock(return_value=payload))
    first = request_data(tmp_path)
    path = next(tmp_path.glob("*.json"))
    original_cache = path.read_bytes()
    for identifier in SERIES.values():
        payload["observations"][1][identifier] = {"v": "broken"}
    fallback = request_data(tmp_path, refresh=True)
    assert fallback.source == "stale cache"
    assert fallback.fetched_at == first.fetched_at
    assert path.read_bytes() == original_cache
    pd.testing.assert_frame_equal(fallback.frame, first.frame)


def test_future_observation_rejected_even_with_future_requested_end(payload, tmp_path, monkeypatch):
    payload["observations"][0]["d"] = "2026-11-02"
    monkeypatch.setattr(client, "_request", Mock(return_value=payload))
    with pytest.raises(DataUnavailable, match="outside the requested date range"):
        client.load_data("2026-10-26", "2027-01-01", cache_dir=tmp_path)


def test_out_of_bounds_blank_date_also_rejected(payload, tmp_path, monkeypatch):
    payload["observations"].append({"d": "2026-10-25"})
    monkeypatch.setattr(client, "_request", Mock(return_value=payload))
    with pytest.raises(DataUnavailable, match="outside the requested date range"):
        request_data(tmp_path)

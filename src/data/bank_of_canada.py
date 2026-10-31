"""Bank of Canada Valet client with validated, atomic, provenance-bearing caches."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from .validation import SERIES, DataUnavailable, parse_observations

BASE_URL = "https://www.bankofcanada.ca/valet"
OBSERVATIONS_URL = f"{BASE_URL}/observations/{','.join(SERIES.values())}/json"
SOURCE_URL = "https://www.bankofcanada.ca/rates/interest-rates/canadian-bonds/"
CACHE_TTL = timedelta(hours=6)
TIMEOUT = (5, 30)  # bounded connect and read timeout, in seconds
MAX_ATTEMPTS = 3
CACHE_VERSION = 1


@dataclass
class DataBundle:
    frame: pd.DataFrame
    source: str
    fetched_at: str
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def latest_date(self) -> pd.Timestamp:
        return self.frame.index.max()


def _now() -> datetime:
    return datetime.now(UTC)


def _iso_date(value: str | date, name: str) -> str:
    try:
        return date.fromisoformat(str(value)).isoformat()
    except (ValueError, TypeError) as exc:
        raise DataUnavailable(f"{name} must be a valid YYYY-MM-DD date.") from exc


def _digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _request(params: dict) -> dict:
    """Retry only transient failures, never silently replace a series."""
    failure = "Unknown network error"
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = requests.get(OBSERVATIONS_URL, params=params, timeout=TIMEOUT)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise DataUnavailable("Valet returned a non-object JSON response.")
            return payload
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            failure = f"HTTP {status}"
            if status is not None and status < 500 and status not in {408, 429}:
                break
        except (requests.RequestException, ValueError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
        if attempt < MAX_ATTEMPTS - 1:
            time.sleep(0.5 * 2**attempt)
    raise DataUnavailable(f"Bank of Canada request failed after bounded retries ({failure}).")


def _bundle(envelope: dict, request: dict, source: str) -> DataBundle:
    """Validate the entire cached envelope, including its query coverage."""
    if envelope.get("version") != CACHE_VERSION or envelope.get("request") != request:
        raise DataUnavailable("Cache version or requested date coverage does not match.")
    if envelope.get("url") != OBSERVATIONS_URL:
        raise DataUnavailable("Cache source URL does not match official benchmark request.")
    try:
        fetched = datetime.fromisoformat(envelope["fetched_at"])
        if fetched.tzinfo is None or fetched.utcoffset() != timedelta(0):
            raise ValueError("Fetch timestamp must be UTC")
        if fetched > _now() + timedelta(minutes=5):
            raise ValueError("Fetch timestamp is in the future")
        payload = envelope["payload"]
        if envelope.get("sha256") != _digest(payload):
            raise ValueError("Payload checksum mismatch")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise DataUnavailable(f"Invalid cache provenance or contents: {exc}.") from exc
    frame, warnings = parse_observations(payload)
    if frame.index.min().date() - date.fromisoformat(request["start_date"]) > timedelta(days=7):
        warnings.append(
            f"Returned coverage starts {frame.index.min().date().isoformat()}, later than "
            f"requested {request['start_date']}; available history is limited by the source."
        )
    upper = min(request["end_date"] or fetched.date().isoformat(), fetched.date().isoformat())
    # Validate all dates, including non-observation rows omitted by the parser.
    for row in payload["observations"]:
        if not request["start_date"] <= row["d"] <= upper:
            raise DataUnavailable(
                "Response contains observations outside the requested date range."
            )
    return DataBundle(
        frame=frame,
        source=source,
        fetched_at=fetched.isoformat(),
        warnings=warnings,
        metadata={
            "provider": "Bank of Canada Valet API",
            "source_url": SOURCE_URL,
            "observations_url": OBSERVATIONS_URL,
            "series": dict(SERIES),
            "series_details": payload["seriesDetail"],
            "units": "percent",
            "frequency": "Business daily",
            "requested_start": request["start_date"],
            "requested_end": request["end_date"],
            "coverage_start": frame.index.min().date().isoformat(),
            "coverage_end": frame.index.max().date().isoformat(),
            "payload_sha256": envelope["sha256"],
            "terms_url": "https://www.bankofcanada.ca/terms/",
            "curve_type": "Government of Canada benchmark bond yields; not a zero-coupon curve",
        },
    )


def _save_cache(path: Path, envelope: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(envelope, handle, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_data(
    start_date: str = "2000-01-01",
    end_date: str | None = None,
    refresh: bool = False,
    cache_dir: Path = Path(".cache/valet"),
) -> DataBundle:
    """Load validated yields; use six-hour cache or explicitly labeled stale fallback.

    Cache keys include the exact requested interval. Omitted ``end_date`` means
    latest available and shares a stable cache key across days. Fetch time measures
    retrieval freshness; ``latest_date`` separately measures observation freshness.
    No expected business-day calendar is imposed, and no missing yields are filled.
    """
    start = _iso_date(start_date, "start_date")
    end = _iso_date(end_date, "end_date") if end_date is not None else None
    if start > (end or _now().date().isoformat()):
        raise DataUnavailable("start_date must not be after end_date (or today).")
    request = {"start_date": start, "end_date": end}
    key = _digest(request)[:24]
    path = Path(cache_dir) / f"benchmark_yields_{key}.json"
    cached, cache_warning = None, None
    try:
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                envelope = json.load(handle)
            if not isinstance(envelope, dict):
                raise DataUnavailable("Cached envelope is not an object.")
            cached = _bundle(envelope, request, "cache")
    except (OSError, ValueError, TypeError, DataUnavailable) as exc:
        cache_warning = f"Ignored unusable local cache: {exc}"
    if cached and not refresh:
        age = _now() - datetime.fromisoformat(cached.fetched_at)
        if age <= CACHE_TTL:
            return cached
    try:
        params = {"start_date": start}
        if end:
            params["end_date"] = end
        payload = _request(params)
        envelope = {
            "version": CACHE_VERSION,
            "request": request,
            "url": OBSERVATIONS_URL,
            "fetched_at": _now().isoformat(),
            "payload": payload,
            "sha256": _digest(payload),
        }
        bundle = _bundle(envelope, request, "live")
    except (DataUnavailable, ValueError, TypeError) as exc:
        if cached:
            cached.source = "stale cache"
            cached.warnings.append(
                f"Live refresh failed; showing previously validated cached observations. {exc}"
            )
            return cached
        detail = f" {cache_warning}" if cache_warning else ""
        raise DataUnavailable(f"No usable official data or validated cache. {exc}{detail}") from exc
    if cache_warning:
        bundle.warnings.append(cache_warning)
    try:
        _save_cache(path, envelope)
    except OSError as exc:
        bundle.warnings.append(f"Live data loaded, but local cache could not be saved: {exc}")
    return bundle

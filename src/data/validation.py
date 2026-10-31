"""Strict structural validation for official daily benchmark yield observations."""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np
import pandas as pd

SERIES = {f"{year}Y": f"BD.CDN.{year}YR.DQ.YLD" for year in (2, 3, 5, 7, 10)}
TENORS = list(SERIES)


class DataUnavailable(RuntimeError):
    """No validated official observations can be provided for the requested range."""


def validate_metadata(payload: dict) -> dict:
    """Require the exact benchmark series and check tenor, frequency and units."""
    if not isinstance(payload, Mapping):
        raise DataUnavailable("Valet response must be a JSON object.")
    details = payload.get("seriesDetail")
    if not isinstance(details, Mapping):
        raise DataUnavailable("Valet response is missing series metadata.")
    for tenor, identifier in SERIES.items():
        detail = details.get(identifier)
        if not isinstance(detail, Mapping):
            raise DataUnavailable(
                f"Official benchmark metadata missing for {tenor} ({identifier})."
            )
        label = str(detail.get("label", "")).lower()
        description = str(detail.get("description", "")).lower()
        if not re.search(rf"\b{tenor[:-1]}[- ]year\b", label):
            raise DataUnavailable(f"Unexpected benchmark maturity in metadata for {identifier}.")
        if "benchmark" not in label or "yield" not in label:
            raise DataUnavailable(f"Unexpected benchmark label for {identifier}.")
        if "government of canada" not in description:
            raise DataUnavailable(f"Unexpected issuer metadata for {identifier}.")
        metadata = detail.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise DataUnavailable(f"Malformed units metadata for {identifier}.")
        if str(metadata.get("unit", "")).lower() not in {"percent", "percentage", "%"}:
            raise DataUnavailable(f"Expected percentage yield units for {identifier}.")
        if str(metadata.get("frequency", "")).lower() not in {"business daily", "daily"}:
            raise DataUnavailable(f"Expected daily frequency for {identifier}.")
    return dict(details)


def parse_observations(payload: dict) -> tuple[pd.DataFrame, list[str]]:
    """Parse percentage yields without interpolation or forward filling.

    Source-blank rows are omitted: these do not constitute observations (e.g. holidays).
    An all-null row containing invalid supplied values rejects the response; dropping
    a corrupted observation would silently alter subsequent observation-based horizons.
    A row with any actual yield is retained, including a weekend if supplied by the
    source. Missing tenors remain NaN. Invalid numeric cells are diagnosed as NaN;
    malformed dates, duplicate dates and inconsistent metadata reject the response.
    """
    validate_metadata(payload)
    observations = payload.get("observations")
    if not isinstance(observations, list) or not observations:
        raise DataUnavailable("Valet returned no observations for the requested period.")
    records, dates, warnings = [], [], []
    invalid = {tenor: 0 for tenor in TENORS}
    for observation in observations:
        if not isinstance(observation, Mapping):
            raise DataUnavailable("Malformed Valet observation row.")
        raw_date = observation.get("d")
        if not isinstance(raw_date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_date):
            raise DataUnavailable(f"Invalid observation date: {raw_date!r}.")
        try:
            dates.append(pd.to_datetime(raw_date, format="%Y-%m-%d", errors="raise"))
        except (ValueError, TypeError, OverflowError) as exc:
            raise DataUnavailable(f"Invalid observation date: {raw_date!r}.") from exc
        row = {}
        row_invalid = False
        for tenor, identifier in SERIES.items():
            cell = observation.get(identifier)
            value = cell.get("v") if isinstance(cell, Mapping) else None
            if value is None or value == "":
                row[tenor] = np.nan
                if cell is not None and not isinstance(cell, Mapping):
                    invalid[tenor] += 1
                    row_invalid = True
                continue
            try:
                number = float(value) if not isinstance(value, bool) else np.nan
            except (TypeError, ValueError, OverflowError):
                number = np.nan
            if not np.isfinite(number):
                invalid[tenor] += 1
                row_invalid = True
                number = np.nan
            row[tenor] = number
        if row_invalid and all(np.isnan(value) for value in row.values()):
            raise DataUnavailable(
                f"All yields missing/invalid on {raw_date}, with corrupt supplied values; "
                "refusing to remove an invalid observation from the analysis calendar."
            )
        records.append(row)
    index = pd.DatetimeIndex(dates, name="date")
    if index.has_duplicates:
        raise DataUnavailable("Duplicate observation dates in Valet response; data rejected.")
    frame = pd.DataFrame(records, index=index, columns=TENORS, dtype=float).sort_index()
    null_rows = int(frame.isna().all(axis=1).sum())
    frame = frame.dropna(how="all")
    if frame.empty:
        raise DataUnavailable("No numeric benchmark yield observations in Valet response.")
    if null_rows:
        warnings.append(f"Omitted {null_rows} dates with no yield observations in any tenor.")
    for tenor in TENORS:
        if invalid[tenor]:
            warnings.append(
                f"{tenor}: {invalid[tenor]} invalid/non-finite values treated as missing."
            )
        missing = int(frame[tenor].isna().sum())
        if missing:
            warnings.append(
                f"{tenor}: {missing} missing observations retained; no forward filling."
            )
    return frame, warnings

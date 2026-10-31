"""Historical reference statistics: strictly preceding observation windows."""

import numpy as np
import pandas as pd


def validate_series(series: pd.Series) -> pd.Series:
    """Reject ambiguous temporal inputs; preserve missing rows and percentage/bps units."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("A DatetimeIndex is required.")
    if (
        series.index.hasnans
        or not series.index.is_monotonic_increasing
        or not series.index.is_unique
    ):
        raise ValueError("Observation dates must be valid, unique, and increasing.")
    values = pd.to_numeric(series, errors="raise").astype(float)
    if np.isinf(values).any():
        raise ValueError("Infinite observations are invalid; represent unavailable values as NaN.")
    return values


def rolling_statistics(series: pd.Series, window: int = 60) -> pd.DataFrame:
    """Use t-window...t-1, sample std (ddof=1), and prior-window percentile midranks.

    Every window must have `window` valid observations. NaN rows count as periods
    and invalidate the window instead of stretching it backward. Percentile is
    100 * (count(prior < current) + 0.5 * count(prior == current)) / window.
    """
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 2:
        raise ValueError("window must be an integer of at least 2 observations.")
    values = validate_series(series)
    prior = values.shift(1).rolling(window, min_periods=window)
    result = pd.DataFrame(
        {"spread_bps": values, "mean_bps": prior.mean(), "std_bps": prior.std(ddof=1)}
    )
    result["deviation_bps"] = values - result["mean_bps"]
    # An exactly constant reference cannot support a standardized signal.
    result["zscore"] = result["deviation_bps"] / result["std_bps"].where(result["std_bps"] > 0)
    percentile = np.full(len(values), np.nan)
    array = values.to_numpy()
    for i in range(window, len(array)):
        reference = array[i - window : i]
        if np.isfinite(array[i]) and np.isfinite(reference).all():
            percentile[i] = (
                100.0
                * (np.sum(reference < array[i]) + 0.5 * np.sum(reference == array[i]))
                / window
            )
    result["percentile"] = percentile
    return result[["spread_bps", "mean_bps", "std_bps", "zscore", "percentile", "deviation_bps"]]

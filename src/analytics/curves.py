"""Benchmark term spreads and conservative two-point curve classifications."""

import numpy as np
import pandas as pd

from src.analytics.yields import to_bps


def term_spreads(frame: pd.DataFrame) -> pd.DataFrame:
    """Longer minus shorter benchmark yield, in bps; missing tenors stay missing."""
    result = pd.DataFrame(index=frame.index)
    for name, short, long in (("2s10s", "2Y", "10Y"), ("2s5s", "2Y", "5Y"), ("5s10s", "5Y", "10Y")):
        result[name] = (
            to_bps(frame[long] - frame[short]) if short in frame and long in frame else np.nan
        )
    return result


def classify_curve(short_change_bps: float, long_change_bps: float) -> str:
    """Bull/bear requires both yields to fall/rise; slope is long minus short change."""
    short, long = short_change_bps, long_change_bps
    if not (np.isfinite(short) and np.isfinite(long)):
        return "Mixed/indeterminate"
    slope_change = long - short
    if short * long <= 0 or np.isclose(slope_change, 0.0, atol=1e-10, rtol=0):
        return "Mixed/indeterminate"
    regime = "Bull" if short < 0 else "Bear"
    shape = "steepening" if slope_change > 0 else "flattening"
    return f"{regime} {shape}"

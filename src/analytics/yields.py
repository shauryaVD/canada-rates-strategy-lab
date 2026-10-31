"""Yield levels are percentage units; differences are reported in basis points."""

import numpy as np
import pandas as pd


def to_bps(value):
    """Convert percentage-point changes to basis points (3.65 - 3.20 -> 45)."""
    return value * 100.0


def movement_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare the final row with prior rows, never filling or skipping missing values."""
    result = pd.DataFrame(index=frame.columns)
    result.index.name = "tenor"
    result["current"] = frame.iloc[-1] if len(frame) else np.nan
    result["prior"] = frame.iloc[-2] if len(frame) > 1 else np.nan
    for lag in (1, 5, 20):
        result[f"change_{lag}_bps"] = (
            to_bps(frame.iloc[-1] - frame.iloc[-lag - 1]) if len(frame) > lag else np.nan
        )
    return result

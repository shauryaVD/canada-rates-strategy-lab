"""Historical spread-event study, deliberately not a bond-return or P&L model."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analytics.relative_value import rolling_statistics, validate_series

SIGNAL_COLUMNS = ["date", "end_date", "spread_bps", "zscore", "direction", "status"]
OUTCOME_COLUMNS = [
    "date",
    "end_date",
    "spread_bps",
    "zscore",
    "direction",
    "forward_change_bps",
    "convergence_bps",
    "hit",
]
METRICS = [
    "evaluated_signals",
    "hit_rate_pct",
    "mean_forward_change_bps",
    "median_forward_change_bps",
    "mean_convergence_bps",
    "median_convergence_bps",
]


@dataclass
class BacktestResult:
    """CSV-ready audit trail, evaluated outcomes, summary, unconditional baseline, metadata."""

    signals: pd.DataFrame
    outcomes: pd.DataFrame
    summary: pd.DataFrame
    baseline: pd.DataFrame
    parameters: dict


def _metrics(changes: np.ndarray, convergence: np.ndarray) -> dict:
    if not len(changes):
        return {name: 0 if name == "evaluated_signals" else np.nan for name in METRICS}
    return {
        "evaluated_signals": len(changes),
        "hit_rate_pct": float(100 * np.mean(convergence > 0)),
        "mean_forward_change_bps": float(np.mean(changes)),
        "median_forward_change_bps": float(np.median(changes)),
        "mean_convergence_bps": float(np.mean(convergence)),
        "median_convergence_bps": float(np.median(convergence)),
    }


def run_backtest(
    series: pd.Series,
    window: int = 60,
    threshold: float = 1.5,
    horizon: int = 10,
    start=None,
    end=None,
    non_overlapping: bool = True,
    sample: str = "test",
    train_fraction: float = 0.7,
) -> BacktestResult:
    """Evaluate prior-window statistical extremes over h subsequent observation rows.

    Split is based on ALL dates in the selected analysis range, before signal or
    future-data eligibility filtering. Earlier data may warm up historical windows.
    Training events ending on/after the first test date are purged. Test windows
    may use earlier training history, as it was available at signal time. The end
    date strictly bounds all outcomes. An event with any missing spread on its
    inclusive path is excluded. Nonoverlap greedily accepts t >= previous endpoint.

    Baseline: all complete forward windows beginning in the same selected sample,
    irrespective of z-score availability; overlapping windows are retained and
    labelled. No significance or independence claims are implied.
    """
    values = validate_series(series)
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer.")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive.")
    if not np.isfinite(train_fraction) or not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if sample not in {"train", "test", "all"}:
        raise ValueError("sample must be train, test, or all.")
    if start is not None and end is not None and pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("start must not be after end.")
    stats = rolling_statistics(values, window)
    selected_mask = np.ones(len(values), dtype=bool)
    if start is not None:
        selected_mask &= values.index >= pd.Timestamp(start)
    if end is not None:
        selected_mask &= values.index <= pd.Timestamp(end)
    positions = np.flatnonzero(selected_mask)
    split_offset = int(np.floor(len(positions) * train_fraction))
    split_position = int(positions[split_offset]) if len(positions) else None
    sample_positions = positions
    if sample == "train":
        sample_positions = positions[:split_offset]
    elif sample == "test":
        sample_positions = positions[split_offset:]
    final_position = int(positions[-1]) if len(positions) else -1
    raw = values.to_numpy()
    zscores = stats["zscore"].to_numpy()
    signal_rows, outcome_rows, baseline_changes = [], [], []
    last_endpoint = -1

    def exclusion(position: int) -> str | None:
        endpoint = position + horizon
        if endpoint > final_position:
            return "excluded_no_future"
        if sample == "train" and endpoint >= split_position:
            return "excluded_cross_split"
        if not np.isfinite(raw[position : endpoint + 1]).all():
            return "excluded_missing_path"
        return None

    for position in sample_positions:
        position = int(position)
        endpoint = position + horizon
        reason = exclusion(position)
        if reason is None:
            baseline_changes.append(float(raw[endpoint] - raw[position]))
        zscore = zscores[position]
        if not np.isfinite(zscore) or abs(zscore) < threshold:
            continue
        direction = "high" if zscore > 0 else "low"
        sign = -1 if direction == "high" else 1
        if reason is None and non_overlapping and position < last_endpoint:
            reason = "excluded_overlap"
        row = {
            "date": values.index[position],
            "end_date": values.index[endpoint] if endpoint <= final_position else pd.NaT,
            "spread_bps": raw[position],
            "zscore": zscore,
            "direction": direction,
            "status": reason or "evaluated",
        }
        signal_rows.append(row)
        if reason is not None:
            continue
        change = float(raw[endpoint] - raw[position])
        convergence = sign * change
        outcome_rows.append(
            {key: row[key] for key in OUTCOME_COLUMNS[:5]}
            | {
                "forward_change_bps": change,
                "convergence_bps": convergence,
                "hit": convergence > 0,
            }
        )
        last_endpoint = endpoint

    signals = pd.DataFrame(signal_rows, columns=SIGNAL_COLUMNS)
    outcomes = pd.DataFrame(outcome_rows, columns=OUTCOME_COLUMNS)
    summaries = []
    for direction in ("all", "high", "low"):
        subset = outcomes if direction == "all" else outcomes[outcomes["direction"] == direction]
        valid = (
            len(signals) if direction == "all" else int((signals["direction"] == direction).sum())
        )
        summaries.append(
            {
                "direction": direction,
                "valid_signals": valid,
                **_metrics(
                    subset["forward_change_bps"].to_numpy(dtype=float),
                    subset["convergence_bps"].to_numpy(dtype=float),
                ),
            }
        )
    summary = pd.DataFrame(summaries).set_index("direction")
    changes = np.asarray(baseline_changes, dtype=float)
    baseline = pd.DataFrame(
        [
            {"direction": direction, **_metrics(changes, sign * changes)}
            for direction, sign in (("high", -1), ("low", 1))
        ]
    ).set_index("direction")
    parameters = {
        "window": window,
        "threshold": threshold,
        "horizon": horizon,
        "start": values.index[positions[0]].date().isoformat() if len(positions) else None,
        "end": values.index[positions[-1]].date().isoformat() if len(positions) else None,
        "sample": sample,
        "train_fraction": train_fraction,
        "split_date": values.index[split_position].date().isoformat()
        if split_position is not None
        else None,
        "non_overlapping": non_overlapping,
        "reference_window": "Prior observations only; full valid window; sample std ddof=1",
        "units": "basis-point spread changes, not returns or profit",
        "baseline_overlap": "All eligible unconditional windows; overlapping windows retained",
        "missing_path_rule": "Exclude any event containing missing spread observations",
        "hit_rule": "Strictly positive direction-adjusted convergence; zero is a miss",
        "overlap_rule": "Greedy chronological; next signal may share the previous endpoint",
    }
    return BacktestResult(signals, outcomes, summary, baseline, parameters)

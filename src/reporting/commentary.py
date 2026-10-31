"""Deterministic research commentary from supplied observations, never a forecast."""

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

SOURCE_URL = "https://www.bankofcanada.ca/valet-api-how-to/"
TENORS = ("2Y", "3Y", "5Y", "7Y", "10Y")
SPREADS = {"2s10s": ("2Y", "10Y"), "2s5s": ("2Y", "5Y"), "5s10s": ("5Y", "10Y")}
DISCLAIMER = (
    "Independent personal research only; not investment advice. "
    "Yield changes and spread convergence are not bond returns or trading P&L."
)


@dataclass(frozen=True)
class Brief:
    """Display-ready report shared by Markdown and PDF exports."""

    report_date: str
    as_of: str
    period: str
    snapshot: tuple[tuple[str, ...], ...]
    sections: tuple[tuple[str, str], ...]
    parameters: dict[str, Any]
    source: str
    fetched_at: str


def _number(value: Any, decimals: int = 2, signed: bool = False) -> str:
    if value is None or not np.isfinite(float(value)):
        return "unavailable"
    return f"{float(value):+.{decimals}f}" if signed else f"{float(value):.{decimals}f}"


def _date(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _spread(row: pd.Series, name: str) -> float:
    short, long = SPREADS[name]
    return (row.get(long, np.nan) - row.get(short, np.nan)) * 100


def _change(frame: pd.DataFrame, tenor: str, lag: int) -> float:
    if tenor not in frame or len(frame) <= lag:
        return np.nan
    return (frame[tenor].iloc[-1] - frame[tenor].iloc[-1 - lag]) * 100


def _evidence(backtest: Any) -> str:
    summary = getattr(backtest, "summary", pd.DataFrame())
    if summary.empty or "all" not in summary.index:
        return (
            "No evaluated historical evidence is available for the selected sample and parameters."
        )
    total = summary.loc["all"]
    evaluated = int(total.get("evaluated_signals", 0))
    valid = int(total.get("valid_signals", 0))
    text = f"{valid} valid signals; {evaluated} evaluated events. "
    if not evaluated:
        return text + "No completed forward outcomes are available; no success rate is inferred."
    text += (
        f"Movement in the hypothesized direction: {_number(total.get('hit_rate_pct'))}%. "
        "Directional evidence (raw forward spread changes): "
    )
    details = []
    for direction, hypothesis in (("high", "narrowing"), ("low", "widening")):
        if direction not in summary.index:
            continue
        row = summary.loc[direction]
        n = int(row.get("evaluated_signals", 0))
        if n:
            details.append(
                f"{direction} signals / {hypothesis}: n={n}, "
                f"mean {_number(row.get('mean_forward_change_bps'), signed=True)} bps, "
                f"median {_number(row.get('median_forward_change_bps'), signed=True)} bps"
            )
    text += "; ".join(details) + ". "
    baseline = getattr(backtest, "baseline", pd.DataFrame())
    if not baseline.empty:
        for direction, hypothesis in (("high", "narrowing"), ("low", "widening")):
            if direction in baseline.index:
                row = baseline.loc[direction]
                text += (
                    f"Unconditional {hypothesis} baseline: "
                    f"n={int(row.get('evaluated_signals', 0))}, "
                    f"directional rate {_number(row.get('hit_rate_pct'))}%. "
                )
    return text + "Descriptive historical outcomes; no causal or profitability inference."


def build_brief(
    frame: pd.DataFrame,
    statistics: pd.DataFrame,
    backtest: Any,
    *,
    spread_name: str = "2s10s",
    parameters: dict[str, Any] | None = None,
    source: str = "live",
    fetched_at: str = "",
    report_date: Any = None,
) -> Brief:
    """Create a concise note using the final row, never silently substituting stale yields.

    Input yields are percentage points, indexed by observation date. Statistics must
    use the same index and lagged historical reference windows. Report calculations
    preserve missing values and measure lags in rows of the supplied observation set.
    """
    if spread_name not in SPREADS:
        raise ValueError(f"Unsupported spread: {spread_name}")
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Research brief requires unique observation dates")
    latest = frame.iloc[-1] if not frame.empty else pd.Series(dtype=float)
    as_of = _date(frame.index[-1]) if not frame.empty else "unavailable"
    merged_parameters = dict(getattr(backtest, "parameters", {}) or {})
    merged_parameters.update(parameters or {})
    merged_parameters.setdefault("spread", spread_name)
    merged_parameters.setdefault(
        "input_data_start", _date(frame.index[0]) if not frame.empty else "unavailable"
    )
    merged_parameters.setdefault("input_data_end", as_of)
    analysis_start = merged_parameters.get("analysis_start") or merged_parameters.get("start")
    analysis_end = merged_parameters.get("analysis_end") or merged_parameters.get("end")
    period = (
        f"{analysis_start} to {analysis_end}"
        if analysis_start and analysis_end
        else f"{_date(frame.index[0])} to {as_of}"
        if not frame.empty
        else "unavailable"
    )
    rows = tuple(
        (
            tenor,
            _number(latest.get(tenor, np.nan)),
            _number(_change(frame, tenor, 1), signed=True),
            _number(_change(frame, tenor, 5), signed=True),
        )
        for tenor in TENORS
    )
    snapshot = (
        "; ".join(f"{name}: {_number(_spread(latest, name), signed=True)} bps" for name in SPREADS)
        + ". Changes use prior and five-prior published observation rows; missing endpoints remain unavailable."
    )
    slope = _spread(latest, "2s10s")
    shape = (
        "unavailable"
        if not np.isfinite(slope)
        else "upward sloping"
        if slope > 0
        else "inverted"
        if slope < 0
        else "flat"
    )
    curve = f"The 2-to-10-year segment is {shape}. "
    for lag in (5, 20):
        change = slope - _spread(frame.iloc[-lag - 1], "2s10s") if len(frame) > lag else np.nan
        if np.isfinite(change):
            label = "steepened" if change > 0 else "flattened" if change < 0 else "was unchanged"
            curve += (
                f"It {label} by {_number(abs(change))} bps versus "
                f"{_date(frame.index[-lag - 1])} ({lag} observations earlier). "
            )
        else:
            curve += f"The {lag}-observation comparison is unavailable. "
    curve += "Connected benchmark issue yields are not a fitted zero-coupon curve."
    stat = (
        statistics.loc[frame.index[-1]]
        if not frame.empty and frame.index[-1] in statistics.index
        else pd.Series(dtype=float)
    )
    zscore = stat.get("zscore", np.nan)
    rv = (
        f"{spread_name}: {_number(_spread(latest, spread_name), signed=True)} bps; "
        f"prior-window mean {_number(stat.get('mean_bps'))} bps; "
        f"standard deviation {_number(stat.get('std_bps'))} bps; "
        f"z-score {_number(zscore, signed=True)}; "
        f"historical percentile {_number(stat.get('percentile'), 1)}%; "
        f"deviation {_number(stat.get('deviation_bps'), signed=True)} bps. "
    )
    threshold = merged_parameters.get("threshold", 2.0)
    if np.isfinite(zscore):
        label = (
            "above the positive"
            if zscore >= threshold
            else "below the negative"
            if zscore <= -threshold
            else "within the selected"
        )
        rv += f"The observation is {label} z-score threshold ({threshold:g}). "
    else:
        rv += "A usable lagged reference window is unavailable; no statistical label is assigned. "
    rv += "Statistical deviation does not establish economic mispricing or a trade recommendation."
    risks = (
        "Monitor spread deviation, curve direction, data gaps and observation freshness. "
        "Policy expectations, inflation and term premium require separate evidence; no catalyst is inferred. "
        "Regime shifts and parameter selection may limit generalization. Event overlap reduces independence; "
        "the unconditional baseline may overlap even when selected signals do not. "
        "A tradable strategy additionally needs DV01 hedges, carry, roll-down, costs and financing. "
        "Changing parameters after reviewing test results makes the test sample exploratory rather than an untouched holdout."
    )
    return Brief(
        report_date=_date(report_date or date.today()),
        as_of=as_of,
        period=period,
        snapshot=rows,
        sections=(
            ("1. Market Snapshot", snapshot),
            ("2. Yield Curve Analysis", curve),
            ("3. Relative-Value Observation", rv),
            ("4. Historical Evidence", _evidence(backtest)),
            ("5. Watchlist and Risks", risks),
        ),
        parameters=merged_parameters,
        source=source,
        fetched_at=fetched_at or "unavailable",
    )


def parameter_text(brief: Brief) -> str:
    """Preserve every supplied backtest parameter in an inspectable stable order."""
    return "; ".join(f"{key}={value}" for key, value in sorted(brief.parameters.items()))


def render_markdown(brief: Brief) -> str:
    """Export exactly the observations and prose used by the PDF."""
    lines = [
        "# CANADA RATES STRATEGY LAB",
        "## Canadian Rates Monitor",
        f"Report date: {brief.report_date} | Data as of: {brief.as_of}",
        f"Analysis observations: {brief.period}",
        "",
        f"Input data coverage (including statistical warm-up): "
        f"{brief.parameters.get('input_data_start', 'unavailable')} to "
        f"{brief.parameters.get('input_data_end', brief.as_of)}.",
        "",
    ]
    for number, (title, text) in enumerate(brief.sections):
        lines += [f"## {title}", ""]
        if number == 0:
            lines += [
                "| Tenor | Yield (%) | Prior change (bps) | Five-observation change (bps) |",
                "| --- | ---: | ---: | ---: |",
                *("| " + " | ".join(row) + " |" for row in brief.snapshot),
                "",
            ]
        lines += [text, ""]
        if number == 3:
            lines += [f"Parameters: {parameter_text(brief)}", ""]
    lines += [
        f"Source: [Bank of Canada Valet API]({SOURCE_URL}). "
        f"Retrieval mode: {brief.source}. Fetched: {brief.fetched_at}.",
        "",
        DISCLAIMER,
        "",
    ]
    return "\n".join(lines)

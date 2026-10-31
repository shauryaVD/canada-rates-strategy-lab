"""Explicitly synthetic report fixtures, never presented as market evidence."""

from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader

from src.reporting import build_brief, render_markdown, render_pdf


@pytest.fixture
def report_inputs():
    dates = pd.bdate_range("2026-09-01", periods=25)
    frame = pd.DataFrame(
        {
            tenor: np.linspace(start, end, len(dates))
            for tenor, start, end in [
                ("2Y", 3.0, 3.2),
                ("3Y", 3.1, 3.3),
                ("5Y", 3.2, 3.4),
                ("7Y", 3.3, 3.5),
                ("10Y", 3.4, 3.65),
            ]
        },
        index=dates,
    )
    stats = pd.DataFrame(
        {
            "mean_bps": [35.0],
            "std_bps": [5.0],
            "zscore": [2.0],
            "percentile": [95.0],
            "deviation_bps": [10.0],
        },
        index=dates[-1:],
    )
    summary = pd.DataFrame(
        {
            "valid_signals": [5, 3, 2],
            "evaluated_signals": [4, 2, 2],
            "hit_rate_pct": [75.0, 100.0, 50.0],
            "mean_forward_change_bps": [-1.0, -3.0, 1.0],
            "median_forward_change_bps": [-0.5, -2.0, 1.0],
        },
        index=["all", "high", "low"],
    )
    baseline = pd.DataFrame(
        {"evaluated_signals": [100, 100], "hit_rate_pct": [40, 60]}, index=["high", "low"]
    )
    params = {
        "window": 20,
        "threshold": 2.0,
        "horizon": 5,
        "start": "2026-09-01",
        "end": "2026-10-05",
        "non_overlapping": True,
        "sample": "test",
        "train_fraction": 0.7,
        "split_date": "2026-09-24",
        "reference_window": "preceding observations, excluding current; ddof=1",
        "units": "basis points",
        "baseline_overlap": True,
    }
    return frame, stats, SimpleNamespace(summary=summary, baseline=baseline, parameters=params)


def test_markdown_contains_dates_units_parameters_and_real_calculations(report_inputs):
    brief = build_brief(*report_inputs, source="synthetic test fixture", report_date="2026-10-31")
    text = render_markdown(brief)
    assert "Data as of: 2026-10-05" in text
    assert "Report date: 2026-10-31" in text
    assert "2s10s: +45.00 bps" in text
    assert "| 2Y | 3.20 | +0.83 | +4.17 |" in text
    assert (
        "| Tenor | Yield (%) | Prior change (bps) | Five-observation change (bps) |\n"
        "| --- | ---: | ---: | ---: |\n"
        "| 2Y | 3.20 | +0.83 | +4.17 |"
    ) in text
    assert "z-score +2.00" in text
    assert "historical percentile 95.0%" in text
    assert "5 valid signals; 4 evaluated events" in text
    assert "mean -3.00 bps" in text
    assert "split_date=2026-09-24" in text
    assert "non_overlapping=True" in text
    assert "https://www.bankofcanada.ca/valet-api-how-to/" in text
    assert "not bond returns or trading P&L" in text


def test_pdf_is_one_page_and_text_matches_numbers_dates(report_inputs):
    brief = build_brief(*report_inputs, source="synthetic test fixture", report_date="2026-10-31")
    result = render_pdf(brief)
    reader = PdfReader(BytesIO(result))
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text()
    for expected in [
        "2026-10-31",
        "2026-10-05",
        "+45.00",
        "3.65",
        "z-score +2.00",
        "4 evaluated events",
    ]:
        assert expected in text
    assert result == render_pdf(brief)


def test_empty_inputs_export_without_fabricated_evidence():
    frame = pd.DataFrame(index=pd.DatetimeIndex([]))
    brief = build_brief(frame, pd.DataFrame(), None, report_date="2026-10-31")
    text = render_markdown(brief)
    assert "Data as of: unavailable" in text
    assert "No evaluated historical evidence is available" in text
    assert "no statistical label is assigned" in text
    assert len(PdfReader(BytesIO(render_pdf(brief))).pages) == 1


def test_missing_latest_yield_and_stats_never_silently_use_stale_values(report_inputs):
    frame, stats, backtest = report_inputs
    frame.loc[frame.index[-1], "10Y"] = np.nan
    stats.index = frame.index[-2:-1]
    brief = build_brief(frame, stats, backtest)
    text = render_markdown(brief)
    assert "| 10Y | unavailable | unavailable | unavailable |" in text
    assert "2s10s: unavailable bps" in text
    assert "z-score unavailable" in text
    assert "Data as of: 2026-10-05" in text


def test_no_completed_outcomes_has_no_success_rate(report_inputs):
    frame, stats, backtest = report_inputs
    backtest.summary["evaluated_signals"] = 0
    text = render_markdown(build_brief(frame, stats, backtest))
    assert "no success rate is inferred" in text
    assert "75.00%" not in text


def test_duplicate_dates_and_unknown_spreads_rejected(report_inputs):
    frame, stats, backtest = report_inputs
    with pytest.raises(ValueError, match="unique observation dates"):
        build_brief(pd.concat([frame, frame.iloc[-1:]]), stats, backtest)
    with pytest.raises(ValueError, match="Unsupported spread"):
        build_brief(frame, stats, backtest, spread_name="invented")


def test_pdf_refuses_silent_overflow(report_inputs):
    brief = build_brief(*report_inputs, parameters={"oversized_custom_parameter": "words " * 5000})
    with pytest.raises(ValueError, match="exceeds one readable page"):
        render_pdf(brief)

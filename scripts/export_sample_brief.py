"""Generate a reproducible real-data demonstration brief outside the app UI."""

from pathlib import Path

import pandas as pd

from src.analytics.backtesting import run_backtest
from src.analytics.curves import term_spreads
from src.analytics.relative_value import rolling_statistics
from src.data import load_data
from src.reporting import build_brief, render_markdown, render_pdf


def main() -> None:
    bundle = load_data()
    history = bundle.frame
    analysis = history.loc[history.index.max() - pd.DateOffset(years=3) :]
    spread_name = "2s10s"
    window = 60
    threshold = 1.5
    horizon = 10
    spreads = term_spreads(history)
    statistics = rolling_statistics(spreads[spread_name], window).loc[analysis.index]
    backtest = run_backtest(
        spreads[spread_name],
        window=window,
        threshold=threshold,
        horizon=horizon,
        start=analysis.index.min(),
        end=analysis.index.max(),
        sample="test",
        non_overlapping=True,
    )
    brief = build_brief(
        history,
        statistics,
        backtest,
        spread_name=spread_name,
        parameters={
            "analysis_start": analysis.index.min().date().isoformat(),
            "analysis_end": analysis.index.max().date().isoformat(),
        },
        source=bundle.source,
        fetched_at=bundle.fetched_at,
    )
    pdf_path = Path("output/pdf/canada_rates_brief.pdf")
    markdown_path = Path("output/markdown/canada_rates_brief.md")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(render_pdf(brief))
    markdown_path.write_text(render_markdown(brief), encoding="utf-8")
    print(f"Wrote {pdf_path} and {markdown_path} for data as of {brief.as_of}.")


if __name__ == "__main__":
    main()

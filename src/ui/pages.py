"""Five research views, sharing validated observations and calculation modules."""

import pandas as pd
import streamlit as st

from src.analytics.backtesting import run_backtest
from src.analytics.curves import classify_curve, term_spreads
from src.analytics.yields import movement_table
from src.reporting import build_brief, render_markdown, render_pdf
from src.ui import charts
from src.ui.components import caption_sample, direction, download_csv, number, plot


def market_overview(frame, history, tenors):
    st.subheader("Market overview")
    st.caption(
        "Published daily observations. Changes are between observation dates, not intraday prices."
    )
    table = movement_table(history)
    spreads = term_spreads(history)
    cols = st.columns(5)
    for i, tenor in enumerate(["2Y", "5Y", "10Y"]):
        row = table.loc[tenor]
        with cols[i]:
            st.metric(
                f"{tenor[:-1]}-year benchmark",
                number(row.current, "%"),
                number(row.change_1_bps, " bps", signed=True),
                delta_color="off",
                border=True,
            )
            st.caption(
                f"{direction(row.change_1_bps)} · 5 obs: {number(row.change_5_bps, ' bps', True)}"
            )
    for i, name in enumerate(["2s10s", "5s10s"], start=3):
        changes = spreads[name].diff()
        five = spreads[name].diff(5)
        with cols[i]:
            st.metric(
                name,
                number(spreads[name].iloc[-1], " bps", True),
                number(changes.iloc[-1], " bps", True),
                delta_color="off",
                border=True,
            )
            st.caption(
                f"{direction(changes.iloc[-1])} · 5 obs: {number(five.iloc[-1], ' bps', True)}"
            )
    st.subheader("Benchmark yield history")
    caption_sample(frame)
    if tenors:
        plot(charts.lines(frame[tenors], "Yield (%)"), "yield_history")
    else:
        st.info("Select at least one maturity in the sidebar to display the yield history.")
    st.subheader("Observed market movements")
    st.caption(
        "Current values use the selected end date. A missing tenor stays unavailable; changes never skip missing rows."
    )
    display = table.rename(
        columns={
            "current": "Current (%)",
            "prior": "Prior (%)",
            "change_1_bps": "1 obs (bps)",
            "change_5_bps": "5 obs (bps)",
            "change_20_bps": "20 obs (bps)",
        }
    )
    st.dataframe(display.style.format("{:.2f}", na_rep="Unavailable"), width="stretch")
    export = frame.rename(columns=lambda c: f"{c}_yield_pct").join(
        term_spreads(frame).add_suffix("_bps")
    )
    download_csv("Download selected observations", export, "canada_rates_observations.csv")


def yield_curve(frame, history, window, spread_name):
    st.subheader("Yield curve & spread analytics")
    st.caption(
        "Benchmark bond yield points connected for visualization. This is not a bootstrapped zero-coupon curve."
    )
    selected_date = st.date_input(
        "Historical curve comparison",
        value=frame.index[0].date(),
        min_value=history.index.min().date(),
        max_value=history.index.max().date(),
        key="curve_date",
    )
    comparison = history.loc[: str(selected_date)]
    curves = {f"Selected end · {history.index[-1]:%d %b %Y}": history.iloc[-1]}
    for lag in [5, 20]:
        if len(history) > lag:
            curves[f"{lag} obs earlier · {history.index[-lag - 1]:%d %b %Y}"] = history.iloc[
                -lag - 1
            ]
    if not comparison.empty:
        curves[f"Historical · {comparison.index[-1]:%d %b %Y}"] = comparison.iloc[-1]
        if comparison.index[-1].date() != selected_date:
            st.caption(
                f"No observation on the requested date. Comparison uses {comparison.index[-1]:%Y-%m-%d}, the preceding published date."
            )
    plot(charts.curve(curves), "curve_points")
    if len(history) >= 2:
        short, long = (history[["2Y", "10Y"]].iloc[-1] - history[["2Y", "10Y"]].iloc[-2]) * 100
        st.markdown(f"**Previous-observation move: {classify_curve(short, long)}**")
        st.caption(
            f"2Y: {number(short, ' bps', True)} | 10Y: {number(long, ' bps', True)} | 2s10s: {number(long - short, ' bps', True)}"
        )
    with st.expander("How the curve classification works"):
        st.write(
            "Bull means both 2Y and 10Y yields fell; bear means both rose. Steepening means 10Y minus 2Y increased; flattening means it decreased. Opposite signs, unchanged endpoints or slope, and missing values are mixed/indeterminate. Yield direction alone is not a bond-return calculation."
        )
    spreads = term_spreads(frame)
    st.subheader("Term spreads")
    plot(charts.lines(spreads, "Spread (bps)", zero=True), "spread_history")
    st.subheader(f"{spread_name} · selected-sample context")
    percentile_bands = st.checkbox("Show 10th / 90th percentile bands", value=True, key="bands")
    selected = spreads[spread_name]
    rolling = selected.rolling(window, min_periods=window).mean()
    fig = charts.lines(
        pd.DataFrame({spread_name: selected, f"{window}-obs trailing mean": rolling}),
        "Spread (bps)",
        zero=True,
        height=330,
    )
    if percentile_bands and selected.notna().any():
        for percentile in [0.1, 0.9]:
            fig.add_hline(
                y=selected.quantile(percentile),
                line_dash="dash",
                line_color="#596c83",
                annotation_text=f"Sample {percentile:.0%}",
            )
    plot(fig, "spread_context")
    st.caption(
        f"Observed range: {number(selected.min(), ' bps')} to {number(selected.max(), ' bps')}. Bands and this descriptive trailing mean use only the selected sample; the mean includes its current observation and is not used to create signals."
    )
    download_csv("Download spread history", spreads.add_suffix("_bps"), "curve_spreads.csv")


def relative_value(frame, stats, spread_name, window, threshold):
    st.subheader("Relative-value research")
    st.caption(
        "A statistical screen of curve spreads. Extremes do not establish economic mispricing."
    )
    latest = stats.iloc[-1]
    cols = st.columns(4)
    for col, label, value, suffix in [
        (cols[0], spread_name, latest.spread_bps, " bps"),
        (cols[1], "Prior-window mean", latest.mean_bps, " bps"),
        (cols[2], "Z-score", latest.zscore, ""),
        (cols[3], "Historical percentile", latest.percentile, "%"),
    ]:
        col.metric(label, number(value, suffix), border=True)
    st.caption(
        f"Prior-window standard deviation: {number(latest.std_bps, ' bps')} | Deviation from mean: {number(latest.deviation_bps, ' bps', True)} | Reference: {window} prior observations, excluding the current date"
    )
    if pd.isna(latest.zscore):
        st.info(
            "A current signal is unavailable: the prior window has insufficient observations, missing values, or zero variance."
        )
    else:
        label = (
            "Above historical range threshold"
            if latest.zscore >= threshold
            else "Below historical range threshold"
            if latest.zscore <= -threshold
            else "Within selected historical range"
        )
        st.markdown(f"**{label}**")
        st.write(
            f"At {frame.index[-1]:%d %b %Y}, {spread_name} is {number(latest.deviation_bps, ' bps', True)} from its preceding {window}-observation mean (z = {latest.zscore:.2f}). This measures statistical deviation, not fair value or an expected return. Policy expectations, inflation and term-premium evidence would be needed to assess a possible explanation."
        )
    plot(
        charts.lines(
            stats[["spread_bps", "mean_bps"]].rename(
                columns={"spread_bps": spread_name, "mean_bps": "Prior-window mean"}
            ),
            "Spread (bps)",
        ),
        "rv_spread",
    )
    plot(charts.zscores(stats, threshold), "rv_zscores")
    plot(
        charts.histogram(stats.spread_bps, "Spread (bps)", current=latest.spread_bps),
        "rv_distribution",
    )
    st.caption(
        "Distribution uses the displayed sample. The reported percentile ranks the current value against the prior rolling reference window, with half-weight for ties."
    )
    events = stats.loc[stats.zscore.abs() >= threshold]
    st.subheader("Historical statistical events")
    if events.empty:
        st.info("No threshold events in this sample with the selected parameters.")
    else:
        st.dataframe(
            events.tail(50)
            .sort_index(ascending=False)
            .style.format("{:.2f}", na_rep="Unavailable"),
            width="stretch",
        )
        st.caption(
            f"Showing the most recent {min(50, len(events))} of {len(events):,} events. Consecutive threshold dates may belong to one persistent episode."
        )
    download_csv("Download relative-value observations", stats, "relative_value.csv")


def backtesting(frame, series, result, options):
    st.subheader("Historical signal backtesting")
    st.caption(
        "Event study of subsequent spread changes, measured in basis points. No portfolio returns or profitability estimates."
    )
    summary = result.summary
    total = summary.loc["all"]
    cols = st.columns(4)
    for col, label, value in [
        (cols[0], "Valid threshold signals", f"{int(total.valid_signals):,}"),
        (cols[1], "Evaluated events", f"{int(total.evaluated_signals):,}"),
        (cols[2], "Direction hit rate", number(total.hit_rate_pct, "%")),
        (cols[3], "Mean signed convergence", number(total.mean_convergence_bps, " bps", True)),
    ]:
        col.metric(label, value, border=True)
    st.info(
        "Chronological split: first 70% of selected dates for training, last 30% for test. Parameters are user-selected; inspecting or tuning on test results makes this an exploratory analysis, not an untouched out-of-sample performance claim."
    )
    with st.expander("Active assumptions and evaluation boundaries", expanded=False):
        st.json(result.parameters)
        st.write(
            "Signals use the completed observation at t and the preceding rolling window. Forward change is spread(t+h) minus spread(t). Positive signed convergence means narrowing after a high signal or widening after a low signal; zero is a miss. Evaluation excludes unavailable future observations, missing paths and training events crossing the test boundary. Non-overlap keeps the earliest qualifying event, then waits until its endpoint. Baseline uses all eligible forward windows, including overlaps; event counts and sampling differ. Neither series supports independent-observation significance claims."
        )
        st.write(
            "Trading implementation would require DV01 exposure, curve hedging, carry, roll-down, execution costs and financing. These are not modeled."
        )
    plot(
        charts.signal_chart(frame.pipe(term_spreads)[options["spread_name"]], result.signals),
        "backtest_signals",
    )
    st.subheader("Observed outcomes by signal direction")
    st.dataframe(summary.style.format(precision=2, na_rep="Unavailable"), width="stretch")
    st.caption(
        "High signals hypothesize narrowing; low signals hypothesize widening. The pooled raw mean can cancel opposite directions; signed convergence aligns the two hypotheses."
    )
    if result.outcomes.empty:
        st.warning(
            "No evaluated events for this selection. Extend the sample, lower the threshold, or use a shorter horizon. No results have been imputed."
        )
    else:
        plot(
            charts.histogram(
                result.outcomes.forward_change_bps,
                "Forward spread change (bps)",
                name="Evaluated events",
            ),
            "outcome_distribution",
        )
        st.dataframe(
            result.outcomes.tail(25).sort_values("date", ascending=False),
            width="stretch",
            hide_index=True,
        )
    st.subheader("Unconditional historical baseline")
    st.dataframe(result.baseline.style.format(precision=2, na_rep="Unavailable"), width="stretch")
    st.caption(
        "Baseline is the same selected evaluation period and horizon without a z-score filter. It applies each direction's hypothesis to all eligible dates. Windows overlap, so this is context, not a matched independent control group."
    )
    st.subheader("Horizon sensitivity")
    rows = []
    for horizon in [5, 10, 20]:
        comparison = (
            result
            if horizon == options["horizon"]
            else run_backtest(
                series,
                window=options["window"],
                threshold=options["threshold"],
                horizon=horizon,
                start=frame.index[0],
                end=frame.index[-1],
                sample=options["sample"],
                non_overlapping=options["non_overlapping"],
            )
        )
        row = comparison.summary.loc["all"].to_dict()
        row["Horizon (observations)"] = horizon
        rows.append(row)
    horizons = pd.DataFrame(rows).set_index("Horizon (observations)")
    st.dataframe(horizons.style.format(precision=2, na_rep="Unavailable"), width="stretch")
    st.caption(
        "Horizon changes alter future-data eligibility and, with non-overlap enabled, the accepted event sequence. Counts need not match."
    )
    st.subheader("Export the study")
    c1, c2, c3 = st.columns(3)
    with c1:
        download_csv("Signal observations", result.signals, "signals.csv", index=False)
        download_csv(
            "Study parameters",
            pd.DataFrame(list(result.parameters.items()), columns=["parameter", "value"]),
            "parameters.csv",
            index=False,
        )
    with c2:
        download_csv("Forward outcomes", result.outcomes, "forward_outcomes.csv", index=False)
        download_csv("Historical baseline", result.baseline, "baseline.csv")
    with c3:
        download_csv("Summary statistics", summary, "backtest_summary.csv")
        download_csv("Horizon comparison", horizons, "horizon_comparison.csv")


def research_brief(frame, history, stats, result, options, bundle):
    st.subheader("Automated rates strategy brief")
    st.caption(
        "Deterministic commentary from the selected observations and parameters. No LLM, forecasts or inferred macroeconomic catalysts."
    )
    brief = build_brief(
        history,
        stats,
        result,
        spread_name=options["spread_name"],
        parameters={
            **result.parameters,
            "analysis_start": str(frame.index[0].date()),
            "analysis_end": str(frame.index[-1].date()),
        },
        source=bundle.source,
        fetched_at=bundle.fetched_at,
    )
    markdown = render_markdown(brief)
    pdf = render_pdf(brief)
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download one-page PDF", pdf, "canada_rates_brief.pdf", "application/pdf")
    c2.download_button("Download Markdown", markdown, "canada_rates_brief.md", "text/markdown")
    with c3:
        download_csv(
            "Download underlying yields",
            history.rename(columns=lambda c: f"{c}_yield_pct"),
            "brief_yields_with_warmup.csv",
        )
    st.caption(
        "Underlying data includes pre-sample warm-up history needed to reproduce lagged statistics; the report's analysis window is listed in its parameters."
    )
    with st.container(border=True):
        st.markdown(markdown)

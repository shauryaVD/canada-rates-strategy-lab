"""Streamlit entry point for Canada Rates Strategy Lab."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from src.analytics.backtesting import run_backtest
from src.analytics.curves import term_spreads
from src.analytics.relative_value import rolling_statistics
from src.data import DataUnavailable, load_data
from src.ui import pages
from src.ui.components import CSS

st.set_page_config(page_title="Canada Rates Strategy Lab", page_icon="📈", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="eyebrow">CANADA / FIXED INCOME</div>', unsafe_allow_html=True)
    st.header("Research workspace")
    page = st.radio(
        "Navigation",
        ["Market Overview", "Yield Curve", "Relative Value", "Backtesting", "Research Brief"],
        label_visibility="collapsed",
        key="page",
    )
    st.divider()
    refresh = st.button("Refresh official data", width="stretch", key="refresh")

with st.spinner("Loading and validating Bank of Canada benchmark observations..."):
    try:
        bundle = load_data(refresh=refresh)
    except DataUnavailable as exc:
        st.title("Canada Rates Strategy Lab")
        st.error(f"Official benchmark data is unavailable. {exc}")
        st.info(
            "Check your network connection and refresh. A validated real-data cache will be used when available; the application never creates substitute market observations."
        )
        st.stop()

full = bundle.frame
if full.empty:
    st.error("No usable observations are available for this request.")
    st.stop()
latest = full.index.max()
with st.sidebar:
    st.subheader("Observation window")
    preset = st.selectbox(
        "History",
        ["3 months", "6 months", "1 year", "3 years", "All available", "Custom"],
        index=3,
        key="preset",
    )
    offsets = {
        "3 months": pd.DateOffset(months=3),
        "6 months": pd.DateOffset(months=6),
        "1 year": pd.DateOffset(years=1),
        "3 years": pd.DateOffset(years=3),
    }
    start = (
        max(full.index.min(), latest - offsets[preset]) if preset in offsets else full.index.min()
    )
    end = latest
    if preset == "Custom":
        dates = st.date_input(
            "Analysis dates",
            value=(start.date(), end.date()),
            min_value=full.index.min().date(),
            max_value=latest.date(),
            key="analysis_dates",
        )
        if len(dates) != 2:
            st.info("Choose both a start and an end date.")
            st.stop()
        start, end = (pd.Timestamp(d) for d in dates)
    tenors = st.multiselect(
        "Yield chart maturities", list(full.columns), default=["2Y", "5Y", "10Y"], key="tenors"
    )
    st.subheader("Research parameters")
    spread_name = st.selectbox("Curve spread", ["2s10s", "2s5s", "5s10s"], key="spread")
    window = st.selectbox("Prior observations", [20, 60, 120], index=1, key="window")
    threshold = st.select_slider(
        "Z-score threshold", options=[1.0, 1.5, 2.0, 2.5, 3.0], value=1.5, key="threshold"
    )
    horizon = st.selectbox("Forward observations", [5, 10, 20], index=1, key="horizon")
    sample_label = st.selectbox(
        "Evaluation sample",
        ["Test (last 30%)", "Train (first 70%)", "All (descriptive)"],
        key="sample",
    )
    sample = {"Test (last 30%)": "test", "Train (first 70%)": "train", "All (descriptive)": "all"}[
        sample_label
    ]
    non_overlapping = st.checkbox("Exclude overlapping events", value=True, key="non_overlap")
    st.divider()
    st.caption(
        "Research only. Public daily data. No bank affiliation, trade execution or investment advice."
    )

frame = full.loc[start:end]
if frame.empty:
    st.warning(
        "No published observation dates fall inside the selected range. Select a wider period."
    )
    st.stop()
history = full.loc[:end]
st.markdown('<div class="eyebrow">DAILY BENCHMARK RESEARCH</div>', unsafe_allow_html=True)
st.title("Canada Rates Strategy Lab")
st.caption("Canadian government bond yields · curve dynamics · historical signal evidence")
status_cols = st.columns([1, 1, 2])
status_cols[0].markdown(f"**Data as of {latest:%d %b %Y}**")
status_cols[1].markdown(f"**{bundle.source.upper()}** · validated observations")
status_cols[2].markdown(
    f"Selected end: **{frame.index[-1]:%d %b %Y}** · {spread_name} · {window}-obs reference · ±{threshold:.1f}σ · {horizon}-obs horizon"
)
today = pd.Timestamp(datetime.now(ZoneInfo("America/Toronto")).date())
age = (today - latest.normalize()).days
st.caption(
    f"Bank of Canada Valet API | Retrieved {bundle.fetched_at} | Latest observation age: {age} calendar days"
)
if bundle.source == "stale cache":
    st.warning(
        "The API refresh failed. Displaying a previously validated real-data snapshot; this is not a live retrieval."
    )
if age > 4:
    st.warning(
        f"Latest published observation is {age} calendar days old. Check the source and refresh before drawing current-market conclusions."
    )
with st.expander("Data provenance & quality"):
    st.write(
        "Yields are percent per annum for selected benchmark bonds. No forward-filling or invented holiday observations. Observation-count horizons follow published rows, not calendar-day intervals."
    )
    st.markdown(
        "[Official benchmark data](https://www.bankofcanada.ca/rates/interest-rates/canadian-bonds/) · [Valet documentation](https://www.bankofcanada.ca/valet-api-how-to/)"
    )
    st.write(
        f"Available coverage: {full.index.min():%Y-%m-%d} to {latest:%Y-%m-%d}; {len(full):,} observation dates."
    )
    for warning in bundle.warnings:
        st.warning(warning)
    st.json(bundle.metadata)
st.divider()

options = dict(
    spread_name=spread_name,
    window=window,
    threshold=threshold,
    horizon=horizon,
    sample=sample,
    non_overlapping=non_overlapping,
)
series = term_spreads(history)[spread_name]
stats = rolling_statistics(series, window).loc[frame.index]
if page == "Market Overview":
    pages.market_overview(frame, history, tenors)
elif page == "Yield Curve":
    pages.yield_curve(frame, history, window, spread_name)
elif page == "Relative Value":
    pages.relative_value(frame, stats, spread_name, window, threshold)
else:
    result = run_backtest(
        series,
        window=window,
        threshold=threshold,
        horizon=horizon,
        start=frame.index[0],
        end=frame.index[-1],
        sample=sample,
        non_overlapping=non_overlapping,
    )
    if page == "Backtesting":
        pages.backtesting(frame, series, result, options)
    else:
        pages.research_brief(frame, history, stats, result, options, bundle)

st.divider()
st.caption(
    "Source: Bank of Canada Valet API. Benchmark yields are not zero-coupon rates. Statistical deviations and historical convergence are not evidence of tradable profits."
)

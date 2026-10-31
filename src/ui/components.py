"""Shared presentation, with units and missing-data formatting kept explicit."""

import pandas as pd
import streamlit as st

CSS = """
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px;}
h1 {font-size: 2rem !important; letter-spacing: -.04em; font-weight: 650 !important;}
h2 {font-size: 1.35rem !important; letter-spacing: -.02em;}
h3 {font-size: 1.05rem !important;}
[data-testid="stMetricValue"] {font-size: 1.75rem; font-variant-numeric: tabular-nums;}
[data-testid="stMetricLabel"] {color: #a5b5ca;}
[data-testid="stSidebar"] {border-right: 1px solid #263246;}
[data-testid="stSidebar"] .block-container {padding-top: 2rem;}
.eyebrow {color: #50d2bc; font-size: .72rem; letter-spacing: .16em; font-weight: 650;}
</style>
"""


def number(value, suffix="", signed=False, digits=2):
    if pd.isna(value):
        return "Unavailable"
    return f"{value:+.{digits}f}{suffix}" if signed else f"{value:.{digits}f}{suffix}"


def direction(value):
    if pd.isna(value):
        return "Insufficient history"
    return "Higher" if value > 0 else "Lower" if value < 0 else "Unchanged"


def plot(fig, key):
    st.plotly_chart(
        fig,
        width="stretch",
        key=key,
        config={"displaylogo": False, "toImageButtonOptions": {"format": "png"}},
    )


def csv_bytes(frame, *, index=True):
    return frame.to_csv(index=index, float_format="%.8f").encode("utf-8")


def download_csv(label, frame, filename, *, index=True, key=None):
    st.download_button(label, csv_bytes(frame, index=index), filename, "text/csv", key=key)


def caption_sample(frame):
    st.caption(
        f"Selected sample: {frame.index.min():%d %b %Y} to {frame.index.max():%d %b %Y} "
        f"| {len(frame):,} observation dates | No interpolation or forward-filling"
    )

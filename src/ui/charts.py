"""Charts of observed yields and statistical outcomes; no generated market data."""

import pandas as pd
import plotly.graph_objects as go

COLORS = ["#50d2bc", "#729bff", "#e3b86e", "#c7a1f3", "#91b5ca"]
SOURCE = "Source: Bank of Canada Valet API | Daily benchmark observations"


def style(fig, y_title, *, height=410, x_title="Observation date"):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", size=12, color="#c9d5e5"),
        margin=dict(l=16, r=20, t=36, b=65),
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0),
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#253247", zeroline=False)
    fig.add_annotation(
        text=SOURCE,
        xref="paper",
        yref="paper",
        x=0,
        y=-0.19,
        showarrow=False,
        font=dict(size=10, color="#91a3bb"),
        xanchor="left",
    )
    return fig


def lines(frame, y_title, *, zero=False, height=410):
    fig = go.Figure()
    for i, column in enumerate(frame.columns):
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[column],
                name=str(column),
                mode="lines",
                connectgaps=False,
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                hovertemplate=f"%{{y:.2f}} {y_title}<extra>{column}</extra>",
            )
        )
    if zero:
        fig.add_hline(y=0, line_color="#697d96", line_dash="dot")
    return style(fig, y_title, height=height)


def curve(curves):
    fig = go.Figure()
    for i, (label, row) in enumerate(curves.items()):
        fig.add_trace(
            go.Scatter(
                x=[int(t[:-1]) for t in row.index],
                y=row.values,
                name=label,
                mode="lines+markers",
                connectgaps=False,
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                hovertemplate="%{x}-year: %{y:.2f}%<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_xaxes(tickvals=[2, 3, 5, 7, 10])
    return style(fig, "Yield (%)", x_title="Benchmark maturity (years)")


def zscores(stats, threshold):
    fig = lines(
        stats[["zscore"]].rename(columns={"zscore": "Z-score"}), "Z-score", zero=True, height=310
    )
    for value in [-threshold, threshold]:
        fig.add_hline(y=value, line_dash="dash", line_color="#e3b86e")
    events = stats[stats.zscore.abs() >= threshold]
    fig.add_trace(
        go.Scatter(
            x=events.index,
            y=events.zscore,
            mode="markers",
            name="Threshold events",
            marker=dict(color="#e3b86e", size=6),
            hovertemplate="Z = %{y:.2f}<extra>Threshold event</extra>",
        )
    )
    return fig


def histogram(series, x_title, *, current=None, name="Historical observations"):
    fig = go.Figure(
        go.Histogram(x=series.dropna(), nbinsx=35, name=name, marker_color=COLORS[0], opacity=0.8)
    )
    if current is not None and pd.notna(current):
        fig.add_vline(
            x=current, line_color=COLORS[2], line_width=2, annotation_text="Selected-date value"
        )
    return style(fig, "Observation count", height=330, x_title=x_title)


def signal_chart(series, signals):
    fig = lines(series.to_frame("Spread"), "Spread (bps)")
    for direction, color, symbol in [
        ("high", "#e3b86e", "triangle-down"),
        ("low", "#729bff", "triangle-up"),
    ]:
        events = signals.loc[signals.direction == direction]
        fig.add_trace(
            go.Scatter(
                x=events.date,
                y=events.spread_bps,
                mode="markers",
                name=f"{direction.title()} signals",
                marker=dict(color=color, size=8, symbol=symbol),
                customdata=events[["zscore", "status"]],
                hovertemplate="%{y:.2f} bps<br>Z: %{customdata[0]:.2f}<br>%{customdata[1]}<extra></extra>",
            )
        )
    return fig

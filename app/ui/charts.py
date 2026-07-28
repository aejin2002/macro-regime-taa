"""Plotly chart builders -- pastel palette, light theme only. Every
function takes already-computed data (a Series/DataFrame/dict) and
returns a Figure; no strategy computation happens here."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from ui.theme import COLORS, REGIME_LINE_COLOR, plotly_axes, plotly_layout

RANGE_BUTTONS = [
    dict(count=1, label="1M", step="month", stepmode="backward"),
    dict(count=3, label="3M", step="month", stepmode="backward"),
    dict(count=6, label="6M", step="month", stepmode="backward"),
    dict(count=1, label="1Y", step="year", stepmode="backward"),
    dict(count=2, label="2Y", step="year", stepmode="backward"),
    dict(step="all", label="ALL"),
]


def _title(text: str) -> dict:
    return dict(text=text, font=dict(size=15, color=COLORS["text"]))


def _with_range_selector(
    fig: go.Figure, *, rangeslider: bool = False, initial_days: int | None = None
) -> None:
    fig.update_xaxes(
        rangeselector=dict(
            buttons=RANGE_BUTTONS,
            bgcolor=COLORS["card_bg"],
            bordercolor=COLORS["border"],
            font=dict(color=COLORS["text_secondary"], size=11),
            activecolor=COLORS["sky_light"],
        ),
        rangeslider=dict(visible=rangeslider, thickness=0.06, bgcolor=COLORS["sky_light"]),
        type="date",
    )
    if initial_days:
        x_values = [
            x for trace in fig.data if trace.x is not None for x in trace.x if x is not None
        ]
        if x_values:
            end = max(x_values)
            start = pd.Timestamp(end) - pd.Timedelta(days=initial_days)
            fig.update_xaxes(range=[start, end])


def allocation_donut(weights: dict[str, float], title: str) -> go.Figure:
    series = pd.Series(weights).sort_values(ascending=False)
    series = series[series.abs() > 1e-6]
    palette = ["#79BCE8", "#BFE8DD", "#DCEFFA", "#F7E6AE", "#4A9FD5", "#A8D8CB", "#C9DCEA", "#E6D4A8"]
    fig = go.Figure(
        go.Pie(
            labels=series.index,
            values=series.values,
            hole=0.55,
            marker=dict(colors=palette[: len(series)], line=dict(color=COLORS["card_bg"], width=2)),
            texttemplate="%{label}<br>%{percent}",
            textposition="outside",
            hovertemplate="%{label}: %{value:.1%}<extra></extra>",
        )
    )
    fig.update_layout(**plotly_layout(title=_title(title), height=320, showlegend=False))
    return fig


def allocation_bar(weights: dict[str, float], title: str) -> go.Figure:
    series = pd.Series(weights).sort_values(ascending=True)
    series = series[series.abs() > 1e-6]
    fig = go.Figure(
        go.Bar(
            x=series.values,
            y=series.index,
            orientation="h",
            marker_color=COLORS["sky"],
            text=[f"{v:.1%}" for v in series.values],
            textposition="outside",
        )
    )
    fig.update_layout(**plotly_layout(title=_title(title), height=320))
    fig.update_xaxes(tickformat=".0%")
    plotly_axes(fig)
    return fig


def signal_history_chart(
    series: pd.Series,
    title: str,
    *,
    thresholds: list[tuple[float, str]] | None = None,
    tick_suffix: str = "",
    initial_days: int | None = None,
) -> go.Figure:
    data = pd.to_numeric(series, errors="coerce").dropna()
    fig = go.Figure()
    if not data.empty:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.values,
                mode="lines",
                line=dict(color=COLORS["accent"], width=2),
                fill="tozeroy",
                fillcolor="rgba(74,159,213,0.08)",
                name=title,
                hovertemplate="%{x|%Y-%m-%d}: %{y}<extra></extra>",
            )
        )
    for level, label in thresholds or []:
        fig.add_hline(
            y=level, line_dash="dash", line_color=COLORS["risk"], line_width=1.2, annotation_text=label
        )
    fig.update_layout(**plotly_layout(title=_title(title), height=260, showlegend=False))
    plotly_axes(fig)
    if tick_suffix:
        fig.update_yaxes(ticksuffix=tick_suffix)
    _with_range_selector(fig, initial_days=initial_days)
    return fig


def regime_shaded_chart(regime_series: pd.Series, title: str = "Regime History") -> go.Figure:
    clean = regime_series.dropna()
    fig = go.Figure()
    if clean.empty:
        fig.update_layout(**plotly_layout(title=_title(title), height=280))
        return fig

    spell_id = (clean != clean.shift(1)).cumsum()
    for _, spell in clean.groupby(spell_id):
        regime = spell.iloc[0]
        color = REGIME_LINE_COLOR.get(regime, COLORS["border"])
        fig.add_vrect(x0=spell.index.min(), x1=spell.index.max(), fillcolor=color, opacity=0.22, line_width=0)

    for regime, color in REGIME_LINE_COLOR.items():
        if regime in set(clean):
            fig.add_trace(
                go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=10, color=color), name=regime)
            )
    fig.update_layout(**plotly_layout(title=_title(title), height=280))
    fig.update_yaxes(visible=False)
    plotly_axes(fig)
    _with_range_selector(fig, rangeslider=True)
    return fig


def crisis_shaded_line_chart(
    value_series: pd.Series, crisis_mode: pd.Series | None, title: str, *, initial_days: int | None = None
) -> go.Figure:
    fig = go.Figure()
    data = pd.to_numeric(value_series, errors="coerce").dropna()
    fig.add_trace(
        go.Scatter(
            x=data.index, y=data.values, mode="lines", line=dict(color=COLORS["accent"], width=2), name=title
        )
    )
    if crisis_mode is not None:
        on = crisis_mode.reindex(data.index) == "ON"
        if on.any():
            spell_id = (on != on.shift(1)).cumsum()
            for _, spell in pd.DataFrame({"on": on}).groupby(spell_id):
                if bool(spell["on"].iloc[0]):
                    fig.add_vrect(
                        x0=spell.index.min(),
                        x1=spell.index.max(),
                        fillcolor=COLORS["risk"],
                        opacity=0.15,
                        line_width=0,
                    )
    fig.update_layout(**plotly_layout(title=_title(title), height=320, showlegend=False))
    plotly_axes(fig)
    _with_range_selector(fig, rangeslider=True, initial_days=initial_days)
    return fig


def nav_chart(
    nav_df: pd.DataFrame, title: str = "Cumulative Return", *, initial_days: int | None = None
) -> go.Figure:
    palette = [COLORS["accent"], COLORS["sky"], "#B79ACC", "#E0A8A0"]
    fig = go.Figure()
    for i, col in enumerate(nav_df.columns):
        fig.add_trace(
            go.Scatter(
                x=nav_df.index,
                y=nav_df[col],
                mode="lines",
                name=col,
                line=dict(width=2.4 if i == 0 else 1.8, color=palette[i % len(palette)]),
            )
        )
    fig.update_layout(**plotly_layout(title=_title(title), height=340, hovermode="x unified"))
    plotly_axes(fig)
    _with_range_selector(fig, initial_days=initial_days)
    return fig


def drawdown_chart(
    drawdown_df: pd.DataFrame, title: str = "Drawdown from Running Peak", *, initial_days: int | None = None
) -> go.Figure:
    palette = [COLORS["accent"], COLORS["sky"], "#B79ACC", "#E0A8A0"]
    fig = go.Figure()
    for i, col in enumerate(drawdown_df.columns):
        fig.add_trace(
            go.Scatter(
                x=drawdown_df.index,
                y=drawdown_df[col],
                mode="lines",
                name=col,
                line=dict(width=2.4 if i == 0 else 1.8, color=palette[i % len(palette)]),
                fill="tozeroy" if i == 0 else None,
                fillcolor="rgba(74,159,213,0.08)" if i == 0 else None,
            )
        )
    fig.update_layout(**plotly_layout(title=_title(title), height=300, hovermode="x unified"))
    fig.update_yaxes(tickformat=".0%")
    plotly_axes(fig)
    _with_range_selector(fig, initial_days=initial_days)
    return fig


def annual_returns_chart(annual_df: pd.DataFrame, title: str = "Annual Returns") -> go.Figure:
    palette = [COLORS["accent"], COLORS["sky"], "#B79ACC", "#E0A8A0"]
    fig = go.Figure()
    for i, col in enumerate(annual_df.columns):
        fig.add_trace(
            go.Bar(
                x=annual_df.index.astype(str),
                y=annual_df[col],
                name=col,
                marker_color=palette[i % len(palette)],
            )
        )
    fig.add_hline(y=0, line_color=COLORS["border"], line_width=1)
    fig.update_layout(**plotly_layout(title=_title(title), height=300, barmode="group"))
    fig.update_yaxes(tickformat=".0%")
    plotly_axes(fig)
    return fig


def intraday_return_bar(returns: dict[str, float | None], title: str = "Today's Move by Asset") -> go.Figure:
    labels, values = [], []
    for name, v in returns.items():
        if v is not None and pd.notna(v):
            labels.append(name)
            values.append(v)
    colors = [COLORS["mint"] if v >= 0 else COLORS["risk"] for v in values]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors))
    fig.add_hline(y=0, line_color=COLORS["border"], line_width=1)
    fig.update_layout(**plotly_layout(title=_title(title), height=280, showlegend=False))
    fig.update_yaxes(tickformat=".1%")
    plotly_axes(fig)
    return fig

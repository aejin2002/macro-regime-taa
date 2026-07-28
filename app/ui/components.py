"""Reusable pastel UI components -- hero, status/signal/rule/snapshot/
allocation/warning cards, and the strategy-flow row. Every function
renders already-computed values passed in by the caller; nothing here
computes strategy output."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.formatters import UNKNOWN_DISPLAY, arrow_for, fmt_date
from ui.theme import STATE_TONE, TONE_BG, TONE_BORDER, TONE_TEXT


def render_hero(title: str, subtitle: str, meta: dict[str, str]) -> None:
    # Built as a single joined string, deliberately with NO leading
    # whitespace and NO blank lines: Streamlit's markdown-to-HTML pass
    # treats a 4+-space-indented or blank-line-separated fragment as a
    # code block, which would render the tags as literal text instead
    # of parsing them as HTML.
    meta_html = " &nbsp;·&nbsp; ".join(f"{k}: <b>{v}</b>" for k, v in meta.items())
    html = (
        '<div class="hero">'
        f'<div class="hero-title">{title}</div>'
        f'<div class="hero-sub">{subtitle}</div>'
        f'<div class="hero-meta">{meta_html}</div>'
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def status_tone_for(state: str | None) -> str:
    return STATE_TONE.get(fmt_status_safe(state), "neutral")


def fmt_status_safe(state: str | None) -> str:
    return state if state in ("ON", "OFF", "UNKNOWN") else "UNKNOWN"


def render_status_card(title: str, value: str, status: str | None, note: str, tone: str) -> None:
    bg, border, text = TONE_BG[tone], TONE_BORDER[tone], TONE_TEXT[tone]
    status_html = f'<div class="sac-status">{status}</div>' if status else ""
    html = (
        f'<div class="sac-card" style="background:{bg}; border-color:{border}; color:{text};">'
        f'<div class="sac-label">{title}</div>'
        f'<div class="sac-value">{value}</div>'
        f"{status_html}"
        f'<div class="sac-note">{note}</div>'
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def status_pill_html(state: str | None) -> str:
    tone = status_tone_for(state)
    bg, border, text = TONE_BG[tone], TONE_BORDER[tone], TONE_TEXT[tone]
    label = fmt_status_safe(state)
    style = f"background:{bg};border-color:{border};color:{text};"
    return f'<span class="status-pill" style="{style}">{label}</span>'


def render_flow(nodes: list[dict]) -> None:
    """`nodes`: [{"title": str, "state": str, "note": str}, ...] rendered
    left-to-right connected by arrows."""
    parts = ['<div class="flow-row">']
    for i, node in enumerate(nodes):
        if i > 0:
            parts.append('<div class="flow-arrow">→</div>')
        parts.append(
            f'<div class="flow-node"><div class="flow-node-title">{node["title"]}</div>'
            f'<div class="flow-node-state">{node["state"]}</div>'
            f'<div class="flow-node-note">{node.get("note", "")}</div></div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_signal_card(
    *,
    name: str,
    current_display: str,
    previous_display: str | None,
    direction: str | None,
    rule_text: str,
    state: str | None,
    observed_date,
    note: str | None = None,
) -> None:
    arrow = arrow_for(direction) if direction else ""
    prev_html = (
        f'<div class="sac-note">Prior: {previous_display} {arrow}</div>' if previous_display else ""
    )
    note_html = f'<div class="sac-note">{note}</div>' if note else ""
    # Only ON/OFF/UNKNOWN-typed signals (gates, shocks) get a status pill --
    # a Growth/Inflation Up/Down card has no such state and must not show a
    # misleading "UNKNOWN" pill just because `state` wasn't one of the three.
    pill_html = ""
    if state is not None:
        pill_html = f'<div style="margin-top:0.6rem;">{status_pill_html(state)}</div>'
    with st.container(border=True):
        html = (
            f'<div class="sac-label">{name}</div>'
            f'<div class="sac-value">{current_display}</div>'
            f"{prev_html}"
            f'<div class="sac-note" style="margin-top:0.5rem;">{rule_text}</div>'
            f"{pill_html}"
            f"{note_html}"
            f'<div class="sac-note" style="margin-top:0.4rem; opacity:0.6;">'
            f"As of {fmt_date(observed_date)}</div>"
        )
        st.markdown(html, unsafe_allow_html=True)


def render_rule_card(title: str, rows: list[tuple[str, str]], caption: str | None = None) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if caption:
            st.caption(caption)
        html = ['<table class="rule-table"><tbody>']
        for label, value in rows:
            html.append(f"<tr><td>{label}</td><td style='text-align:right;'>{value}</td></tr>")
        html.append("</tbody></table>")
        st.markdown("".join(html), unsafe_allow_html=True)


def render_snapshot_board(title: str, items: list[dict], columns_per_row: int = 4) -> None:
    """`items`: [{"label", "value", "delta", "direction", "state"}, ...].
    `direction`/`state` decide favorable-vs-unfavorable framing via the
    caller (this component only renders what it's given)."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        for start in range(0, len(items), columns_per_row):
            row = items[start : start + columns_per_row]
            cols = st.columns(columns_per_row)
            for col, item in zip(cols, row, strict=False):
                with col:
                    st.metric(
                        item["label"],
                        item["value"],
                        delta=item.get("delta"),
                        delta_color=item.get("delta_color", "off"),
                        border=True,
                    )


def render_allocation_stage_card(title: str, weights: dict[str, float] | None, note: str = "") -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if not weights:
            st.caption(UNKNOWN_DISPLAY)
            return
        series = pd.Series(weights).sort_values(ascending=False)
        series = series[series.abs() > 1e-6]
        for asset, w in series.items():
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;font-size:0.85rem;'
                f'padding:0.15rem 0;"><span>{asset}</span><span style="font-weight:700;">'
                f"{w * 100:.1f}%</span></div>",
                unsafe_allow_html=True,
            )
        if note:
            st.caption(note)


def render_warning_card(message: str, *, exc: Exception | None = None) -> None:
    st.warning(message)
    if exc is not None:
        with st.expander("Technical details"):
            st.exception(exc)

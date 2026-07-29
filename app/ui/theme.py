"""Light pastel sky-blue theme -- palette, global CSS, and the shared
Plotly layout template. UI presentation only; no strategy data or
computation lives here.

Layout/pattern inspiration (cards, snapshot boards, range-selector
charts, HTML signal table) came from reviewing a separate local
reference dashboard's structure. No code, strategy rules, tickers, or
data from that project were copied -- only the general layout idea.
"""

from __future__ import annotations

import streamlit as st

COLORS: dict[str, str] = {
    "app_bg": "#F4F9FD",
    "card_bg": "#FFFFFF",
    "sky": "#79BCE8",
    "sky_light": "#DCEFFA",
    "accent": "#4A9FD5",
    # Three-tier navy text hierarchy: primary (headings/values), mid
    # (subtitles), muted (captions/secondary labels). Plotly DATA lines
    # keep their own sky/accent palette -- this hierarchy is text-only.
    "text": "#17365D",
    "text_mid": "#274C77",
    "text_secondary": "#496581",
    "border": "#D8E8F2",
    "mint": "#BFE8DD",
    "warning": "#F7E6AE",
    "risk": "#F3B7B2",
    "amber": "#FFF3D6",
}

# One tone per card "temperature" -- used for both HTML cards (background/
# border) and st.metric-adjacent accents. "risk" is a strongly emphasized
# tone (ON / active-danger states); "reflation" is a distinct warm accent
# for the REFLATION regime specifically, deliberately not reusing "risk"'s
# red so it never reads as a warning/error next to Fast Crisis.
TONE_BG: dict[str, str] = {
    "neutral": COLORS["card_bg"],
    "info": COLORS["sky_light"],
    "good": COLORS["mint"],
    "warn": COLORS["warning"],
    "risk": COLORS["risk"],
    "reflation": "#FFF3D6",
}
TONE_BORDER: dict[str, str] = {
    "neutral": COLORS["border"],
    "info": COLORS["sky"],
    "good": "#8FCFBD",
    "warn": "#E8CE6F",
    "risk": "#E08A82",
    "reflation": "#E6B85C",
}
TONE_TEXT: dict[str, str] = {
    "neutral": COLORS["text"],
    "info": COLORS["text"],
    "good": COLORS["text"],
    "warn": COLORS["text"],
    "risk": "#8C2E27",
    "reflation": "#9A5B00",
}

# One color per regime, used consistently everywhere a regime appears
# (cards, tables, charts). Pastel, not saturated. REFLATION uses its own
# warm "reflation" tone (not "info") so it stands out from the neutral
# sky-blue used elsewhere.
REGIME_TONE: dict[str, str] = {
    "GOLDILOCKS": "good",
    "REFLATION": "reflation",
    "STAGFLATION": "risk",
    "CONTRACTION": "neutral",
    "UNKNOWN": "neutral",
}
REGIME_LINE_COLOR: dict[str, str] = {
    "GOLDILOCKS": "#6FB98F",
    "REFLATION": COLORS["accent"],
    "STAGFLATION": "#D98A82",
    "CONTRACTION": "#8FA3B3",
    "UNKNOWN": "#B8C2CC",
}

# ON is the only strongly-emphasized state; OFF/UNKNOWN stay neutral.
STATE_TONE: dict[str, str] = {"ON": "risk", "OFF": "neutral", "UNKNOWN": "warn"}


def inject_css() -> None:
    c = COLORS
    st.markdown(
        f"""
        <style>
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
        [data-testid="stMain"], section.main, main {{
            background: {c["app_bg"]} !important;
        }}
        .block-container {{
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }}
        [data-testid="stSidebar"] {{
            background: {c["card_bg"]} !important;
            border-right: 1px solid {c["border"]};
        }}
        h1, h2, h3, h4 {{ color: {c["text"]} !important; font-weight: 700; }}
        h1 {{ font-size: 1.9rem; }}
        h2 {{ font-size: 1.25rem; margin-top: 1.4rem; }}
        h3 {{ font-size: 1.05rem; }}
        h4 {{ margin-top: 0.85rem; margin-bottom: 0.4rem; }}
        p, span, label {{ color: {c["text_secondary"]}; }}

        [data-testid="stMetric"] {{
            background: {c["card_bg"]} !important;
            border: 1px solid {c["border"]} !important;
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            box-shadow: 0 1px 3px rgba(35, 54, 77, 0.05);
        }}
        [data-testid="stMetricLabel"] {{ color: {c["text_secondary"]} !important; font-size: 0.8rem; }}
        [data-testid="stMetricValue"] {{ color: {c["text"]} !important; }}

        [data-testid="stVerticalBlockBorderWrapper"] > div {{
            border-color: {c["border"]} !important;
            background: {c["card_bg"]};
            border-radius: 12px;
        }}

        div[data-testid="stDataFrame"] {{ font-size: 0.86rem; }}

        .hero {{ padding: 0.2rem 0 1.1rem 0; }}
        .hero-title {{ font-size: 2.1rem; font-weight: 800; color: {c["text"]}; line-height: 1.15; }}
        .hero-sub {{ font-size: 1rem; color: {c["text_mid"]}; margin-top: 0.2rem; }}
        .hero-meta {{ font-size: 0.85rem; color: {c["text_secondary"]}; margin-top: 0.6rem; }}

        .sac-card {{
            border-radius: 12px;
            padding: 0.95rem 1.15rem;
            border: 1px solid;
            box-shadow: 0 1px 3px rgba(35, 54, 77, 0.05);
            min-height: 118px;
        }}
        .sac-card.sac-compact {{ min-height: 96px; padding: 0.8rem 1rem; }}
        .sac-label {{
            font-size: 0.76rem; font-weight: 700; letter-spacing: 0.04em;
            text-transform: uppercase; opacity: 0.7;
        }}
        .sac-value {{ font-size: 1.7rem; font-weight: 800; margin-top: 0.5rem; line-height: 1.15; }}
        .sac-status {{ font-size: 0.95rem; font-weight: 700; margin-top: 0.5rem; }}
        .sac-note {{ font-size: 0.8rem; margin-top: 0.4rem; line-height: 1.4; opacity: 0.85; }}

        .flow-row {{ display: flex; align-items: stretch; gap: 0.4rem; }}
        .flow-node {{
            flex: 1; border-radius: 12px; padding: 0.9rem 1rem; border: 1px solid {c["border"]};
            background: {c["card_bg"]}; text-align: center;
        }}
        .flow-node-title {{
            font-size: 0.78rem; font-weight: 700; color: {c["text_secondary"]}; text-transform: uppercase;
        }}
        .flow-node-state {{ font-size: 1rem; font-weight: 800; color: {c["text"]}; margin-top: 0.3rem; }}
        .flow-node-note {{ font-size: 0.76rem; color: {c["text_secondary"]}; margin-top: 0.25rem; }}
        .flow-arrow {{
            align-self: center; color: {c["text_secondary"]}; font-size: 1.1rem; padding: 0 0.2rem;
        }}

        .rule-table {{
            width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.88rem;
            border: 1px solid {c["border"]}; border-radius: 10px; overflow: hidden;
        }}
        .rule-table th {{
            background: {c["sky_light"]}; color: {c["text"]}; text-align: left;
            padding: 0.6rem 0.8rem; font-weight: 700;
        }}
        .rule-table td {{ padding: 0.55rem 0.8rem; border-top: 1px solid {c["border"]}; color: {c["text"]}; }}

        .status-pill {{
            display: inline-block; padding: 0.15rem 0.65rem; border-radius: 999px;
            font-size: 0.78rem; font-weight: 700; border: 1px solid;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def plotly_layout(**overrides) -> dict:
    """Shared Plotly layout defaults -- pastel, light-only, no gridline
    clutter. Individual charts pass overrides (height, title, etc.)."""
    base = dict(
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        font=dict(color=COLORS["text_secondary"], size=12),
        margin=dict(l=18, r=18, t=48, b=18),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
        hoverlabel=dict(bgcolor=COLORS["card_bg"], font_color=COLORS["text"], bordercolor=COLORS["border"]),
    )
    base.update(overrides)
    return base


def plotly_axes(fig) -> None:
    fig.update_xaxes(showgrid=False, color=COLORS["text_secondary"], linecolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"], color=COLORS["text_secondary"], zeroline=False)

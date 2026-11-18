"""Visual theme inspired by a modern fleet-management dashboard: light grey canvas, white rounded cards,
dark map hero, one bright green accent, big KPI numbers."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

GREEN = "#3DBE5C"
GREEN_SOFT = "#E7F7EB"
INK = "#111827"
MUTED = "#6B7280"
CANVAS = "#F2F3F5"
CARD = "#FFFFFF"
LINE = "#E5E7EB"
RED = "#E5484D"
DARK_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
DARK_ATTR = "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Inter', system-ui, sans-serif; }}
.stApp {{ background: {CANVAS}; }}
.block-container {{ padding-top: 3.2rem; max-width: 1400px; }}
h1, h2, h3, h4 {{ color: {INK}; font-weight: 700; letter-spacing: -0.01em; }}
/* top brand bar */
.brand {{ display:flex; align-items:center; justify-content:space-between; background:{CARD}; border-radius:18px;
          padding:14px 22px; margin-bottom:14px; box-shadow:0 1px 2px rgba(16,24,40,.05); }}
.brand .logo {{ display:flex; align-items:center; gap:10px; font-weight:700; font-size:20px; color:{INK}; }}
.brand .logo .mark {{ width:30px; height:30px; border-radius:9px; background:{INK}; display:grid; place-items:center; }}
.brand .logo .mark span {{ width:0; height:0; border-left:8px solid transparent; border-right:8px solid transparent;
          border-bottom:13px solid {GREEN}; }}
.brand .sub {{ color:{MUTED}; font-size:13px; }}
.brand .pill {{ background:{GREEN_SOFT}; color:#1E7B37; font-size:12px; font-weight:600; padding:6px 12px; border-radius:999px; }}
/* tabs as pills */
.stTabs [data-baseweb="tab-list"] {{ gap:6px; background:{CARD}; padding:6px; border-radius:999px; width:fit-content;
          box-shadow:0 1px 2px rgba(16,24,40,.05); }}
.stTabs [data-baseweb="tab"] {{ height:38px; padding:0 16px; border-radius:999px; background:transparent; color:{MUTED};
          font-weight:600; font-size:14px; }}
.stTabs [aria-selected="true"] {{ background:{INK} !important; color:#fff !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display:none; }}
/* bordered containers become white cards */
div[data-testid="stVerticalBlockBorderWrapper"] {{ background:{CARD}; border:none !important; border-radius:18px;
          box-shadow:0 1px 2px rgba(16,24,40,.05); }}
div[data-testid="stMetric"] {{ background:{CARD}; border-radius:16px; padding:14px 16px; box-shadow:0 1px 2px rgba(16,24,40,.05); }}
div[data-testid="stMetricValue"] {{ font-weight:700; color:{INK}; }}
div[data-testid="stMetricLabel"] p {{ color:{MUTED}; font-weight:500; }}
div[data-testid="stDataFrame"] {{ border-radius:14px; overflow:hidden; }}
.stButton>button, .stDownloadButton>button {{ border-radius:999px; background:{INK}; color:#fff; border:none; font-weight:600; }}
/* custom KPI + suggestion cards */
.kpi {{ background:{CARD}; border-radius:16px; padding:14px 16px; box-shadow:0 1px 2px rgba(16,24,40,.05); height:100%; }}
.kpi .lbl {{ color:{MUTED}; font-size:13px; font-weight:500; display:flex; gap:6px; align-items:center; }}
.kpi .val {{ color:{INK}; font-size:28px; font-weight:700; margin-top:4px; line-height:1.1; }}
.kpi .tag {{ display:inline-block; margin-top:6px; background:{GREEN_SOFT}; color:#1E7B37; font-size:11px; font-weight:600;
          padding:2px 8px; border-radius:999px; }}
.kpi .tag.bad {{ background:#FDECEC; color:#B42318; }}
.sug {{ background:{CARD}; border-radius:18px; padding:16px 18px; box-shadow:0 1px 2px rgba(16,24,40,.05);
        border:1.5px solid transparent; height:100%; }}
.sug.top {{ border-color:{GREEN}; }}
.sug .rank {{ font-size:12px; font-weight:700; color:#1E7B37; background:{GREEN_SOFT}; padding:3px 9px; border-radius:999px; }}
.sug .name {{ font-size:19px; font-weight:700; color:{INK}; margin:8px 0 2px; }}
.sug .par {{ color:{MUTED}; font-size:13px; margin-bottom:10px; }}
.sug .row {{ display:flex; justify-content:space-between; font-size:13.5px; padding:5px 0; border-top:1px solid {LINE}; }}
.sug .row b {{ color:{INK}; }}
.up {{ color:#1E7B37; font-weight:600; }} .down {{ color:#B42318; font-weight:600; }}
.section-title {{ display:flex; align-items:center; gap:8px; font-weight:700; font-size:16px; color:{INK}; margin:6px 0 8px; }}
.section-title .dot {{ width:8px; height:8px; border-radius:50%; background:{GREEN}; }}
</style>
"""


def apply() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    tmpl = go.layout.Template()
    tmpl.layout = go.Layout(
        font=dict(family="Inter, system-ui, sans-serif", color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=[GREEN, INK, "#9CA3AF", "#60A5FA", RED, "#F59E0B"],
        xaxis=dict(gridcolor=LINE, zeroline=False, linecolor=LINE),
        yaxis=dict(gridcolor=LINE, zeroline=False, linecolor=LINE),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    pio.templates["ajman"] = tmpl
    pio.templates.default = "plotly_white+ajman"


def brand(title: str, subtitle: str, pill: str) -> None:
    st.markdown(f"""<div class="brand"><div><div class="logo"><div class="mark"><span></span></div>{title}</div>
    <div class="sub">{subtitle}</div></div><div class="pill">{pill}</div></div>""", unsafe_allow_html=True)


def kpi(label: str, value: str, tag: str | None = None, bad: bool = False, icon: str = "") -> str:
    t = f'<div class="tag{" bad" if bad else ""}">{tag}</div>' if tag else ""
    return f'<div class="kpi"><div class="lbl">{icon} {label}</div><div class="val">{value}</div>{t}</div>'


BADGES = {"REAL DATA": ("#E7F7EB", "#1E7B37"), "SIMULATED": ("#FFF4E5", "#9A5B00"),
          "ASSUMPTION": ("#FDECEC", "#B42318"), "ESTIMATE": ("#EEF2FF", "#3730A3"), "MODEL": ("#F3F4F6", "#374151")}


def badge(kind: str) -> str:
    bg, fg = BADGES[kind]
    return (f'<span style="background:{bg};color:{fg};font:700 10.5px Inter,sans-serif;letter-spacing:.04em;'
            f'padding:3px 8px;border-radius:999px;margin-left:6px;vertical-align:middle">{kind}</span>')


def section(title: str, *badges: str) -> None:
    b = "".join(badge(k) for k in badges)
    st.markdown(f'<div class="section-title"><span class="dot"></span>{title}{b}</div>', unsafe_allow_html=True)


def _old_section(title: str) -> None:
    st.markdown(f'<div class="section-title"><span class="dot"></span>{title}</div>', unsafe_allow_html=True)


def delta(v: float, fmt: str = "{:+.0f}", good_if_positive: bool = True) -> str:
    good = (v >= 0) == good_if_positive
    return f'<span class="{"up" if good else "down"}">{fmt.format(v)}</span>'

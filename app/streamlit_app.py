"""Ajman Business Survival Score - dashboard.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

from src.features.build_features import SECTORS  # noqa: E402
from src.models.score import best_areas, load_bundle, predict_score  # noqa: E402
from src.viz.risk_map import METRICS, build_growth_map, build_map, build_metric_map  # noqa: E402
from app.pillars import creator_hub, matchmaker, registration_opportunity  # noqa: E402
from app import theme  # noqa: E402
from src.growth.program import district_opportunity, pick_suggestions, registration_view  # noqa: E402
from src.viz.risk_map import build_suggestion_map  # noqa: E402

PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
_D = pd.read_csv(ROOT / "data" / "real" / "districts.csv")
_D = _D[_D["licenses_listed"] > 0].sort_values(["city", "parent", "district"])
AREAS = _D["district"].tolist()
PARENT = dict(zip(_D["district"], _D["parent"]))
area_label = lambda d: f"{d}  ·  {PARENT.get(d, '')}"  # noqa: E731
INK, MUTED = "#1f2937", "#6b7280"

st.set_page_config(page_title="Ajman Digital Growth Platform", page_icon="📈", layout="wide")
theme.apply()


@st.cache_resource
def bundle():
    return load_bundle()


@st.cache_data
def tables():
    return {
        "area": pd.read_csv(PROC / "area_stats.csv"),
        "sector": pd.read_csv(PROC / "sector_forecast.csv"),
        "sat": pd.read_csv(PROC / "saturation_current.csv"),
        "area_sector": pd.read_csv(PROC / "area_sector.csv"),
        "points": pd.read_csv(PROC / "map_points.csv"),
        "metrics": json.loads((REPORTS / "metrics.json").read_text()),
        "annual": pd.read_csv(ROOT / "data" / "real" / "annual.csv"),
        "afz_monthly": pd.read_csv(ROOT / "data" / "real" / "afz_monthly.csv", parse_dates=["month"]),
        "fc_annual": pd.read_csv(PROC / "forecast_annual.csv"),
        "fc_monthly": pd.read_csv(PROC / "forecast_monthly_afz.csv", parse_dates=["month"]),
        "backtest": json.loads((REPORTS / "forecast_backtest.json").read_text()),
        "calib": pd.read_csv(PROC / "calibration_vs_real.csv"),
        "fc_sectors": pd.read_csv(PROC / "forecast_sectors.csv"),
        "fc_districts": pd.read_csv(PROC / "forecast_districts.csv"),
        "fc_afz_type": pd.read_csv(PROC / "forecast_afz_by_type.csv"),
    }


def gauge(score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix": " / 100", "font": {"size": 34}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": INK, "thickness": 0.25},
               "steps": [{"range": [0, 50], "color": "#f5c6c0"}, {"range": [50, 70], "color": "#fbe3b5"},
                         {"range": [70, 100], "color": "#cde8d4"}]}))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=10, b=0))
    return fig


B, T = bundle(), tables()
M = T["metrics"]

theme.brand("Ajman Digital Growth Platform",
            "① Guide new licenses to high-growth districts · ② make Ajman the base for social media marketers · "
            "③ match struggling businesses with Ajman-licensed creators",
            "Real Ajman open data · calibrated model")

tab_fc, tab_ent, tab_hub, tab_match, tab_map, tab_ded, tab_res, tab_method = st.tabs(
    ["📈 Forecast engine", "① Registration Advisor", "② Creator Hub", "③ Growth Matchmaker", "🗺️ Risk map",
     "🏛️ DED insights", "🔎 Research a business", "📐 Method & limits"])

# ---------------------------------------------------------------- Entrepreneur
with tab_ent:
    left, right = st.columns([1, 2.4], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown("#### New license application")
            sector = st.selectbox("Sector", SECTORS, index=SECTORS.index("F&B"))
            area = st.selectbox("Preferred district", AREAS, index=AREAS.index("Nuaimeya 1"), format_func=area_label)
            c1, c2 = st.columns(2)
            capital = c1.number_input("Capital (AED)", 10_000, 50_000_000, 200_000, step=10_000)
            employees = c2.number_input("Employees", 1, 1000, 6)
            c1, c2 = st.columns(2)
            legal = c1.selectbox("Legal form", ["sole establishment", "LLC", "branch", "civil company"])
            premises = c2.selectbox("Premises", ["shop", "office", "warehouse", "factory", "flexi-desk", "home-based"])
            free_zone = st.toggle("Free zone", False)
            st.markdown("**Planned marketing · first 6 months**")
            posts = st.slider("Posts per month", 0, 40, 4)
            c1, c2 = st.columns(2)
            followers = c1.number_input("Instagram followers", 0, 1_000_000, 500, step=100)
            engagement = c2.number_input("Engagement %", 0.0, 10.0, 3.0, 0.5,
                                         help="Likes + comments per follower. Typical 1–5%; bought followers ≈ 0.") / 100
            website = st.toggle("Website", False)
            st.markdown("**Reputation · first 6 months**")
            c1, c2 = st.columns(2)
            reviews = c1.number_input("Google reviews", 0, 1000, 20)
            rating = c2.number_input("Rating ★", 1.0, 5.0, 4.3, 0.1)
            horizon = st.radio("Horizon", [12, 36, 60], index=1, horizontal=True, format_func=lambda m: f"{m // 12} yr")

    biz = {"sector": sector, "area": area, "capital_aed": capital, "employees": employees, "free_zone": free_zone,
           "legal_form": legal, "premises_type": premises, "posts_per_month": posts, "instagram_followers": followers,
           "has_website": website, "google_review_count": reviews, "google_rating": rating,
           "days_since_last_post": 3 if posts else 180, "engagement_rate": engagement}
    r = predict_score(biz, horizon, bundle=B)               # with the applicant's marketing plan
    r_reg = predict_score(registration_view(biz), horizon, bundle=B)  # what is known at registration only
    opp = district_opportunity({k: v for k, v in biz.items() if k != "area"}, B)
    sug, cur = pick_suggestions(opp, area, 3)

    with right:
        with st.container(border=True):
            theme.section(f"Where a {sector} business faces less competition", "REAL DATA")
            st_folium(build_suggestion_map(opp, area, sug["district"].tolist()), height=360,
                      use_container_width=True, returned_objects=[], key="sug_map")
            st.caption("🟢 suggested · 🔴 your choice · hover a district for details. Competitors = real DED "
                       "license-activities of this sector within ~2 km. Districts above saturation 0.30 are never suggested.")
        k = st.columns(4)
        k[0].markdown(theme.kpi("Registration score", f"{r_reg['score']:.0f}<span style='font-size:15px;color:#6B7280'>/100</span>",
                                f"sector · district · size only", bad=r_reg["score"] < 50),
                      unsafe_allow_html=True)
        if cur is not None:
            k[1].markdown(theme.kpi("Competitors nearby", f"{cur['competitors_nearby']:,.0f}",
                                    f"{cur['competitors_per_10k']:.0f} per 10k residents", bad=cur["saturation"] > 0.3),
                          unsafe_allow_html=True)
            k[2].markdown(theme.kpi("District growth 2025→28", f"{cur['growth_pct']:+.0f}%", cur["outlook"],
                                    bad=cur["outlook"] == "Cooling"), unsafe_allow_html=True)
        k[3].markdown(theme.kpi("With your marketing plan", f"{r['score']:.0f}<span style='font-size:15px;color:#6B7280'>/100</span>",
                                f"{r['score'] - r_reg['score']:+.0f} vs registration score",
                                bad=r["score"] < r_reg["score"]), unsafe_allow_html=True)
        st.caption("The registration score uses only what DED knows at licensing (marketing and reviews set to the Ajman "
                   "median), so applicants are not judged on guesses. The marketing-plan score is a what-if. "
                   "Advisory only, never a licensing condition.")

    st.write("")
    theme.section("Suggested districts · less competition, better metrics", "REAL DATA", "MODEL")
    if len(sug) == 0:
        st.info("Your chosen district already has the best mix of competition and metrics for this sector.")
    cols = st.columns(3)
    prio = set(pd.read_csv(PROC / "program_priority_districts.csv")["district"])
    for i, (col, (_, s_)) in enumerate(zip(cols, sug.iterrows())):
        comp_d = (s_["competitors_nearby"] / max(cur["competitors_nearby"], 1) - 1) * 100 if cur is not None else 0
        rows = [("Competitors nearby", f"<b>{s_['competitors_nearby']:,.0f}</b> " + (theme.delta(comp_d, "{:+.0f}%", False) if cur is not None else "")),
                ("Per 10k residents", f"<b>{s_['competitors_per_10k']:.0f}</b>"),
                ("Survival score", f"<b>{s_['score']:.0f}</b> " + (theme.delta(s_['score'] - cur['score']) if cur is not None else "")),
                ("Forecast growth", f"<b>{s_['growth_pct']:+.0f}%</b>"),
                ("Saturation", f"<b>{s_['saturation']:.2f}</b>"),
                ("Room before crowding", f"<b>~{s_['headroom']:,.0f}</b> licenses"),
                ("Opportunity", f"<b>{s_['opportunity']:.0f}</b> " + (theme.delta(s_['opportunity'] - cur['opportunity']) if cur is not None else ""))]
        body = "".join(f'<div class="row"><span>{a_}</span><span>{b_}</span></div>' for a_, b_ in rows)
        badge = " · incentive eligible" if s_["district"] in prio else ""
        col.markdown(f'<div class="sug{" top" if i == 0 else ""}"><span class="rank">#{i + 1} suggestion{badge}</span>'
                     f'<div class="name">{s_["district"]}</div><div class="par">{s_["parent"]} · {int(s_["competitors"])} '
                     f'{sector} license-activities in the district</div>{body}</div>', unsafe_allow_html=True)
    st.caption("Survival differences between nearby districts are small; the suggestions mainly win on competition and "
               "growth. 'Room before crowding' = extra licenses of this sector the catchment can take before saturation "
               "reaches 0.30. DED would rotate suggestions so no district receives more than its room.")

    st.write("")
    a1, a2 = st.columns(2)
    with a1, st.container(border=True):
        theme.section("Survival curve")
        cv = r["curve"]
        fig = px.area(cv, x="month", y="survival", labels={"month": "Months after licensing", "survival": "Still operating"})
        fig.update_traces(line_color="#3DBE5C", fillcolor="rgba(61,190,92,0.15)")
        fig.update_layout(height=230, margin=dict(l=0, r=0, t=0, b=0), yaxis_range=[0, 1], yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
    with a2, st.container(border=True):
        theme.section("What drives this score")
        ct = r["contributions"].copy()
        ct = ct[ct["hazard_change_pct"].abs() >= 0.5]
        fig = px.bar(ct.iloc[::-1], x="hazard_change_pct", y="factor", orientation="h",
                     color=np.where(ct.iloc[::-1]["hazard_change_pct"] > 0, "raises risk", "lowers risk"),
                     color_discrete_map={"raises risk": "#E5484D", "lowers risk": "#3DBE5C"},
                     labels={"hazard_change_pct": "Change in closure risk (%)", "factor": ""})
        fig.update_layout(height=230, margin=dict(l=0, r=0, t=0, b=0), legend_title_text="", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
    w1, w2, w3 = st.columns(3)
    with w1, st.container(border=True):
        theme.section("What if your rating changes?")
        stars = np.round(np.arange(1.0, 5.01, 0.25), 2)
        rows = [{"stars": sv, "reviews": f"{n} reviews",
                 "score": predict_score({**biz, "google_rating": sv, "google_review_count": n}, horizon, B)["score"]}
                for n in (5, 50, 300) for sv in stars]
        fig = px.line(pd.DataFrame(rows), x="stars", y="score", color="reviews",
                      labels={"stars": "Google rating ★", "score": "Score", "reviews": ""},
                      color_discrete_sequence=["#C7CBD1", "#7C8591", "#3DBE5C"])
        fig.add_vline(x=rating, line_dash="dot", line_color=MUTED)
        fig.update_layout(height=220, margin=dict(l=0, r=0, t=0, b=0), yaxis_range=[0, 100], legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width="stretch")
    with w2, st.container(border=True):
        theme.section("What if you post more?")
        ws = [predict_score({**biz, "posts_per_month": p, "days_since_last_post": 3 if p else 180}, horizon, B)["score"]
              for p in range(0, 31, 2)]
        fig = px.line(x=list(range(0, 31, 2)), y=ws, markers=True, labels={"x": "Posts per month", "y": "Score"})
        fig.add_vline(x=posts, line_dash="dot", line_color=MUTED)
        fig.update_traces(line_color="#3DBE5C")
        fig.update_layout(height=220, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, width="stretch")
    with w3, st.container(border=True):
        theme.section("Digital presence breakdown")
        db = r["digital_breakdown"].copy()
        fig = px.bar(db.iloc[::-1], x="contribution", y="component", orientation="h", labels={"contribution": "", "component": ""})
        fig.update_traces(marker_color="#111827")
        fig.update_layout(height=220, margin=dict(l=0, r=0, t=0, b=0), xaxis_range=[0, max(0.45, db["contribution"].max())])
        st.plotly_chart(fig, width="stretch")

    st.divider()
    registration_opportunity(biz, B, r)

with tab_hub:
    creator_hub()

with tab_match:
    matchmaker()

# ---------------------------------------------------------------- Map
with tab_map:
    a = T["area"]
    st.subheader("Ajman business risk map")
    cm1, cm2 = st.columns([1, 3])
    with cm1:
        map_sector = st.selectbox("Sector for saturation layer & popups", SECTORS, index=SECTORS.index("F&B"), key="map_sector")
        st.caption("Real district boundaries (Ajman Municipality). Toggle layers at the top-right; click a district "
                   "for its DED profile: real license counts, new licenses, saturation, most crowded sectors, "
                   "market gap and expected closures. Masfout and Manama are off-screen to the east (zoom out). "
                   "Business dots are simulated, placed in their real district.")
        sel = T["area_sector"][(T["area_sector"]["sector"] == map_sector) & (T["area_sector"]["n"] >= 5)]
        sel = sel.sort_values("saturation", ascending=False)
        st.markdown(f"**{map_sector}: most saturated districts**")
        st.dataframe(sel[["area", "saturation", "n", "per_10k_residents"]].head(6).round(2)
                     .rename(columns={"area": "district", "n": "licenses", "per_10k_residents": "per 10k res."}),
                     hide_index=True, width="stretch")
        st.caption("Masfout/Manama licenses are often used by businesses operating elsewhere (lower fees), "
                   "so their local saturation is less reliable.")
    m = build_map(a, T["points"], T["area_sector"], map_sector)
    with cm2:
        st_folium(m, height=560, use_container_width=True, returned_objects=[])

# ---------------------------------------------------------------- DED
with tab_ded:
    st.subheader("Market map")
    k1, k2, k3 = st.columns([1.3, 1, 2])
    metric = k1.selectbox("Metric", list(METRICS), format_func=METRICS.get, key="ded_metric")
    ded_sector = k2.selectbox("Sector", SECTORS, index=SECTORS.index("F&B"), key="ded_sector",
                              disabled=metric not in ("sector_saturation", "competitors_per_10k"))
    k3.caption("Saturation compares competitors with local customers (residents + visitors) within a ~2 km catchment. "
               "Local sectors (F&B, retail) are judged against nearby demand; B2B sectors (manufacturing, trading) "
               "sell Ajman-wide, so clustering in industrial zones counts only mildly.")
    st_folium(build_metric_map(T["area"], T["area_sector"], metric, ded_sector), height=480,
              use_container_width=True, returned_objects=[], key="ded_map")

    st.subheader("Sector saturation by district (25 districts with most licenses)")
    view = st.radio("Show", ["Saturation (0–1)", "Competitors per 10k residents", "Competitors per customer vs Ajman average"],
                    horizontal=True, key="hm_view")
    col = {"Saturation (0–1)": "saturation", "Competitors per 10k residents": "per_10k_residents",
           "Competitors per customer vs Ajman average": "ratio"}[view]
    top25 = T["area"].sort_values("active", ascending=False)["area"].head(25).tolist()
    hm = T["area_sector"][T["area_sector"]["area"].isin(top25)].pivot(index="sector", columns="area", values=col)[top25]
    fig = px.imshow(hm, color_continuous_scale="Purples", aspect="auto", text_auto=".2f",
                    zmin=0, zmax=(0.5 if col == "saturation" else None))
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), coloraxis_colorbar_title="")
    st.plotly_chart(fig, width="stretch")

    st.markdown("**District profiles**")
    prof = T["area"][T["area"]["active"] > 0][["area", "parent", "population_est", "active", "new_licenses_8m",
                                               "businesses_per_10k", "avg_saturation", "most_saturated",
                                               "best_opportunity", "forecast_closure_rate", "avg_score"]].copy()
    prof.columns = ["District", "Area", "Residents (estimate)", "Licenses in register (real)", "New Nov-25–Jun-26 (real)",
                    "Per 10k res.", "Avg saturation", "Most saturated", "Market gap", "Exp. closure rate 12m", "Avg score"]
    st.dataframe(prof.round(3).sort_values("Licenses in register (real)", ascending=False), hide_index=True, width="stretch")

    d1, d2 = st.columns(2)
    with d1:
        st.subheader("Expected closures, next 12 months")
        sf = T["sector"].sort_values("forecast_closures_12m")
        fig = px.bar(sf, x="forecast_closures_12m", y="sector", orientation="h",
                     hover_data={"active": True, "forecast_closure_rate": ":.1%"},
                     labels={"forecast_closures_12m": "Expected closures", "sector": ""})
        fig.update_traces(marker_color="#c0392b")
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        st.caption("Real license-activity counts per sector x the model's average probability that a business of that "
                   "sector closes in the next 12 months, given its current age.")
    with d2:
        st.subheader("Average score by sector")
        sf = T["sector"].sort_values("avg_score")
        fig = px.bar(sf, x="avg_score", y="sector", orientation="h", labels={"avg_score": "Average score", "sector": ""})
        fig.update_traces(marker_color=INK)
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), xaxis_range=[0, 100])
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- Forecasts
with tab_fc:
    A = T["annual"]
    F = T["fc_annual"]
    BT = T["backtest"]
    theme.section("Ajman DED licenses: history and forecast", "REAL DATA", "MODEL")
    st.latex(r"\text{Active}_{t+1} = \text{Active}_t + \text{New}_{t+1} - \text{Exits}_{t+1},\qquad "
             r"\text{Exits}_{t+1} = \sum_i h_i(\text{age}_i, \text{sector}_i, x_i)\cdot L_{t+1}")
    st.caption("New licenses: average of a log-linear trend (2020–2025) and a damped Holt model. Exits: the survival "
               "model's hazards applied month by month to Ajman's active stock (age and sector mix), at the 2025 exit-risk "
               "level L; bands = scenarios (exit risk keeps falling / returns to 2022–24 level) x entry uncertainty.")
    c1, c2, c3, c4 = st.columns(4)
    f26 = F[F["year"] == 2026].iloc[0]
    c1.metric("Active licenses, end-2026", f"{f26['active']:,.0f}", f"{f26['active'] / A['active'].dropna().iloc[-1] - 1:+.1%} vs 2025")
    c2.metric("New licenses 2026", f"{f26['new']:,.0f}", f"{f26['new'] / A['new'].iloc[-1] - 1:+.1%} vs 2025")
    c3.metric("Exits 2026", f"{f26['exits']:,.0f}", f"{f26['exits'] / A['exits'].dropna().iloc[-1] - 1:+.1%} vs 2025",
              delta_color="inverse")
    c4.metric("Exit rate 2026", f"{f26['exits'] / A['active'].dropna().iloc[-1]:.1%}", "2025: 8.4%", delta_color="off")

    which = st.radio("Series", ["Active licenses", "New licenses", "Exits (cancelled + not renewed)"], horizontal=True)
    col = {"Active licenses": "active", "New licenses": "new", "Exits (cancelled + not renewed)": "exits"}[which]
    hist = A.dropna(subset=[col])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["year"], y=hist[col], mode="lines+markers", name="Real (DED)", line=dict(color=INK)))
    last = hist.iloc[-1]
    fx = [last["year"]] + F["year"].tolist()
    fig.add_trace(go.Scatter(x=fx + fx[::-1], y=[last[col]] + F[f"{col}_hi"].tolist() + F[f"{col}_lo"].tolist()[::-1] + [last[col]],
                             fill="toself", fillcolor="rgba(36,113,163,0.15)", line=dict(width=0), name="Range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fx, y=[last[col]] + F[col].tolist(), mode="lines+markers", name="Forecast",
                             line=dict(color="#2471a3", dash="dash")))
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1),
                      yaxis_title=which)
    st.plotly_chart(fig, width="stretch")
    st.markdown(f"**Backtest** (forecasting 2024–2025 from 2023): active licenses error **{BT['active_mape_model']:.1%}** "
                f"(naive growth {BT['active_mape_naive_growth']:.1%}); new licenses **{BT['new_mape_model']:.1%}** "
                f"(naive {BT['new_mape_naive_last']:.1%}); exits {BT['exits_mape_model']:.0%} (naive {BT['exits_mape_naive_last']:.0%}). "
                "Exits are the hardest part: Ajman's exit rate fell faster in 2025 than the model expected, so the "
                "exit forecast leans cautious.")

    st.subheader("Sector outlook, 2025 → 2028")
    SO = T["fc_sectors"].copy()
    so_avg = 100 * (SO["active_2028"].sum() / SO["active_2025"].sum() - 1)
    fig = px.bar(SO.sort_values("growth_2025_2028_pct"), x="growth_2025_2028_pct", y="sector", orientation="h",
                 color="outlook", color_discrete_map={"Rising": "#2e8b57", "In line": "#aab7c4", "Cooling": "#c0392b"},
                 hover_data={"new_2026": ":,.0f", "exits_2026": ":,.0f", "active_2025": ":,.0f", "active_2028": ":,.0f",
                             "avg_saturation": ":.2f"},
                 labels={"growth_2025_2028_pct": "Growth in active licenses 2025→2028 (%)", "sector": "", "outlook": ""})
    fig.add_vline(x=so_avg, line_dash="dot", line_color=MUTED, annotation_text=f"Ajman {so_avg:+.0f}%")
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")
    st.caption("New licenses per sector = the sector's trend in its share of DED starts (2019–2023, damped) × the "
               "emirate forecast; exits = the survival model's hazards for that sector's stock. Small sectors (tourism) "
               "swing a lot in % terms.")

    st.subheader("District hotspots, 2025 → 2028")
    DO = T["fc_districts"].copy()
    h1, h2 = st.columns([3, 2])
    with h1:
        st_folium(build_growth_map(DO), height=470, use_container_width=True, returned_objects=[], key="growth_map")
    with h2:
        show = DO[DO["outlook"].isin(["Hotspot", "Cooling"])]
        st.markdown("**Fastest growing**")
        st.dataframe(show[show["outlook"] == "Hotspot"][["district", "parent", "active_2025", "new_2026",
                                                           "growth_2025_2028_pct"]].head(8).round(0)
                     .rename(columns={"active_2025": "licenses", "new_2026": "new 2026", "growth_2025_2028_pct": "growth %"}),
                     hide_index=True, width="stretch")
        st.markdown("**Cooling**")
        st.dataframe(show[show["outlook"] == "Cooling"].sort_values("growth_2025_2028_pct")[
                         ["district", "parent", "active_2025", "new_2026", "growth_2025_2028_pct"]].head(6).round(0)
                     .rename(columns={"active_2025": "licenses", "new_2026": "new 2026", "growth_2025_2028_pct": "growth %"}),
                     hide_index=True, width="stretch")
    st.caption("District entries follow each district's trend in its share of DED starts (2021–2023 and 2025; 2024 is "
               "incomplete in the open data). Masfout and Manama licenses are cheaper and often used by businesses "
               "operating elsewhere, so their growth may not mean local activity. Districts with no 2025 starts are "
               "flagged 'Check data' (likely renamed in the register, e.g. Jurf Industrial 3).")

    st.subheader("Ajman Free Zone: new companies per month")
    M_ = T["afz_monthly"]
    FM = T["fc_monthly"]
    fig = go.Figure()
    full = M_[~M_["partial_month"]]
    fig.add_trace(go.Scatter(x=full["month"], y=full["new_companies"], mode="lines+markers", name="Real (AFZ)",
                             line=dict(color=INK)))
    fig.add_trace(go.Scatter(x=list(FM["month"]) + list(FM["month"])[::-1], y=list(FM["hi"]) + list(FM["lo"])[::-1],
                             fill="toself", fillcolor="rgba(36,113,163,0.15)", line=dict(width=0), name="80% range",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=FM["month"], y=FM["new_companies"], mode="lines", name="Forecast",
                             line=dict(color="#2471a3", dash="dash")))
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1),
                      yaxis_title="New companies / month")
    st.plotly_chart(fig, width="stretch")
    lic = [c for c in FM.columns if c.startswith("License:")]
    sp = FM[lic].sum().sort_values(ascending=False).rename(lambda c: c.replace("License: ", ""))
    st.markdown(f"Next 12 months: **{BT['next_12m_total']:,.0f}** new free zone companies "
                f"(last 12 months: {BT['last_12m_total']:,.0f}). Backtest error on the last 6 months: "
                f"{BT['mape_ensemble'] if BT.get('ensemble_used') else BT['mape_ets']:.0%} vs {BT['mape_naive_last']:.0%} for "
                "'same as last month'. Monthly registrations are volatile, so treat the range, not the line.")
    AF = T["fc_afz_type"].copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(y=AF["license_type"], x=AF["last_12m"], name="Last 12 months (real)", orientation="h",
                         marker_color="#aab7c4"))
    fig.add_trace(go.Bar(y=AF["license_type"], x=AF["next_12m"], name="Next 12 months (forecast)", orientation="h",
                         marker_color="#2471a3", text=[f"{g:+.0f}%" for g in AF["growth_pct"]], textposition="outside"))
    fig.update_layout(barmode="group", height=300, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.12),
                      xaxis_title="New free zone companies")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- Research
with tab_res:
    st.subheader("Research a business's digital presence")
    st.caption("An AI agent (Claude + web search) looks up the business's public website, Google listing and social "
               "media, then feeds what it finds into the score. Public business information only; no personal data.")
    from src.research.digital_agent import research_business, to_signals

    r1, r2 = st.columns([2, 1])
    name = r1.text_input("Business name", placeholder="e.g. a café in Al Zorah")
    r_area = r2.selectbox("District", AREAS, key="r_area", format_func=area_label)
    r3, r4, r5 = st.columns(3)
    r_sector = r3.selectbox("Sector", SECTORS, key="r_sector")
    r_cap = r4.number_input("Capital (AED, estimate)", 10_000, 50_000_000, 150_000, step=10_000, key="r_cap")
    r_emp = r5.number_input("Employees (estimate)", 1, 1000, 5, key="r_emp")
    has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    if not has_key:
        st.info("Set the ANTHROPIC_API_KEY environment variable and restart the app to enable live research. "
                "Previously researched (cached) businesses still work.")
    if st.button("Research", type="primary", disabled=not name):
        try:
            with st.spinner("Searching the web…"):
                res = research_business(name, r_area)
            rep = res["report"]
            st.success(f"Match confidence: {rep.get('match_confidence', '?')}" + (" (cached)" if res.get("cached") else ""))
            sig = to_signals(rep)
            rr = predict_score({"sector": r_sector, "area": r_area, "capital_aed": r_cap, "employees": r_emp,
                                **{k: v for k, v in sig.items() if v is not None}}, 36, B)
            k1, k2, k3 = st.columns(3)
            k1.metric("Survival score (indicative)", f"{rr['score']:.0f}")
            k2.metric("Digital Presence Index", f"{rr['digital_presence']:.2f}")
            k3.metric("3-year survival", f"{rr['survival_3y']:.0%}")
            st.markdown("**Findings**")
            st.json({k: v for k, v in rep.items() if k not in ("marketing_observations", "sources")})
            if rep.get("marketing_observations"):
                st.markdown("**Marketing observations**\n" + "\n".join(f"- {o}" for o in rep["marketing_observations"]))
            if rep.get("sources"):
                st.markdown("**Sources**\n" + "\n".join(f"- {s}" for s in rep["sources"]))
            st.caption("Indicative only: the model was trained on activity in a business's first 6 months, "
                       "while this reflects its activity today. Unknown signals use cautious defaults.")
        except Exception as e:  # show the problem instead of crashing the demo
            st.error(f"Research failed: {e}")

# ---------------------------------------------------------------- Method
with tab_method:
    st.subheader("Data: what is real and what is simulated")
    st.markdown(
        "| Used for | Source | Real? |\n|---|---|---|\n"
        "| 79 district boundaries | Ajman Municipality, *Districts area* (data.ajman.ae) | ✅ real |\n"
        "| Licenses per district x sector (saturation, map, business mix) | DED, *Total License Per Activity* (30.5k license-activities, 1,620 activities mapped to 11 sectors) | ✅ real |\n"
        "| Listed licenses & new licenses per district | DED, *License in Ajman (Area)* (61.9k licenses) | ✅ real |\n"
        "| Yearly new / renewed / cancelled / active licenses, GDP | DED file from the organizers (2016–2025) | ✅ real |\n"
        "| Monthly free zone registrations | AFZ file from the organizers (Jan 2024 – Nov 2026) | ✅ real |\n"
        "| Risk by license type and age | Ajman Chamber, *Membership Transactions* 2023 (cancellations by cohort) | ✅ real |\n"
        "| Free zone survival by cohort | Free Zones Authority, *Companies by License Type* (status) | ✅ real (context) |\n"
        "| Housing units per district (resident estimate) | DLRR, *Completed Real Estate Projects* | ✅ real (partial) |\n"
        "| Residents per area, visitor index | Team estimates | ⚠️ estimate |\n"
        "| Individual businesses, their capital, digital presence, reviews | Simulated, calibrated to the real data above | ⚠️ simulated |")
    cal = T["calib"]
    figc = go.Figure()
    figc.add_trace(go.Bar(x=cal["year"], y=cal["real_exit_rate"], name="Real exit rate (DED)", marker_color="#aab7c4"))
    figc.add_trace(go.Scatter(x=cal["year"], y=cal["sim_exit_rate"], name="Simulation", mode="lines+markers",
                              line=dict(color=INK)))
    figc.update_layout(height=260, margin=dict(l=0, r=0, t=30, b=0), yaxis_tickformat=".0%",
                       title="Calibration: simulated vs real yearly exit rate", legend=dict(orientation="h", y=1.15))
    st.plotly_chart(figc, width="stretch")
    st.caption("No public dataset follows individual Ajman businesses from licensing to closure, so firm-level records "
               "are simulated: real district x sector mix, real yearly entries, and hazards tuned until the simulation "
               "reproduces DED's real yearly exits and active stock. This is fitting, not validation: it shows the simulation "
               "is consistent with Ajman's totals, not that its firm-level effects are true. The method is ready to be "
               "retrained on DED's license-level history.")

    st.subheader("Limitations (what we cannot claim yet)")
    st.markdown(
        "1. **Firm-level effects are not measured in Ajman.** The effects of digital presence, reviews and saturation on "
        "survival were set in the simulation and recovered by the model. Hazard ratios and accuracy (C-index) "
        "demonstrate the method, not facts about Ajman.\n"
        "2. **The marketing effect is an assumption** (−20% closure hazard, range 10–30%, x sector relevance). Published "
        "evidence is directional only. The pilot measures it.\n"
        "3. **Exit forecasts are the weakest part.** Backtest error 25% vs 10% for 'same as last year'; new-license "
        "forecasts are strong (1% vs 20%).\n"
        "4. **Residents per district are team estimates**; competitors per 10k residents and saturation inherit them.\n"
        "5. **District growth is fragile**: open data lacks Apr–Dec 2024 starts, some districts were renamed, Masfout/"
        "Manama licenses often serve businesses elsewhere, and small districts swing by large percentages.\n"
        "6. **Sectors come from keyword rules** on 1,620 DED activity names (98% mapped; check below). Counts are "
        "license-activities: a license with two activities counts in both sectors.\n"
        "7. **Creator figures**: the media/marketing share of freelancers (35%), creator prices and capacity are "
        "assumptions; Dubai fees come from third-party summaries.")
    st.subheader("Data request to DED and FZA")
    st.markdown("- **DED:** anonymized license history (issue, renewal, cancellation dates, activity, district) → retrain "
                "the survival model on real outcomes and replace the simulation.\n"
                "- **FZA:** freelance and media licenses by activity (Ajman Free Zone + Ajman Media City) → real creator supply.\n"
                "- **Ajman Statistics Centre:** residents by district → replace resident estimates.\n"
                "- **Pilot:** 200 matched vs 200 comparison businesses, 12 months → the real marketing effect.")
    st.subheader("Units used in this app")
    st.markdown("- **Active licenses** (41,316 end-2025): DED yearly figure, used in forecasts.\n"
                "- **Licenses in DED register** (61,901): DED open-data list by district, includes all listed licenses.\n"
                "- **License-activities** (30,516): DED counts per activity x district; a license can have several activities. "
                "Used for sector mix, competition and saturation.")
    st.subheader("Sector mapping check (top DED activities)")
    from src.data.real import TOP_ACTIVITIES, classify_activity
    st.dataframe(pd.DataFrame([{"DED activity": a, "Licenses": n, "Mapped sector": classify_activity(a)}
                               for a, n in TOP_ACTIVITIES]), hide_index=True, width="stretch")

    st.subheader("The equation")
    st.latex(r"\text{Score}(t, x) = 100 \times S_0(t)^{\exp\left(\beta \cdot x \;+\; \gamma_R R \;+\; \gamma_D D \;+\; \gamma_S\,\text{Saturation}\right)}")
    st.markdown(
        "- $S_0(t)$: baseline survival curve of a Cox proportional-hazards model; $t$ = horizon in months.\n"
        "- $\\beta \\cdot x$: firm basics (sector, capital, employees, free zone, premises, legal form).\n"
        f"- $\\gamma_R = {M['gamma_reputation']:.2f}$, $\\gamma_D = {M['gamma_digital']:.2f}$, "
        f"$\\gamma_S = {M['gamma_saturation']:.2f}$: **estimated from the data** together with $\\beta$, each with a p-value.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Reputation: +1★ trusted rating", f"{(np.exp(M['gamma_reputation']) - 1):+.0%} closure risk")
    c2.metric("Digital presence: 0 → 1", f"{(np.exp(M['gamma_digital']) - 1):+.0%} closure risk")
    c3.metric("Saturation: 0 → 0.5", f"{(np.exp(0.5 * M['gamma_saturation']) - 1):+.0%} closure risk")

    st.subheader("Reputation R: Bayesian average rating")
    st.latex(r"R = \frac{n \cdot \text{rating} + k \cdot \mu}{n + k} - \mu \qquad k = 10,\; \mu = \text{Ajman average rating}")
    st.markdown(f"$\\mu = {M['rating_prior_mean']:.2f}$★. A 1★ rating from 2 reviews barely moves R (it may be one angry "
                "customer); 1★ from 300 reviews moves it almost fully. Same idea as IMDb's top-250 ranking. "
                "R is capped at the range seen in the data so the model never extrapolates.")

    st.subheader("Digital Presence D: learned weights")
    st.latex(r"D = \sum_i w_i \cdot \text{norm}(c_i)")
    wt = pd.DataFrame(list(M["digital_presence_weights"].items()), columns=["component", "learned weight"])
    st.dataframe(wt, hide_index=True)
    st.markdown("Engaged audience = followers × engagement rate. Raw followers can be bought; engaged followers can't, "
                "and the model gives raw counts ~0 weight once engagement is known. Positive review volume = reviews × "
                "(rating − 1)/4, so many reviews only help if they are good. Only signals from the **first 6 months** after "
                "licensing are used, to avoid reverse causality (failing businesses stop posting).")

    st.subheader("Saturation: supply vs demand")
    st.latex(r"\text{ratio}_{a,s} = \frac{\sum_b n_{b,s} K(d_{ab}) \,/\, \sum_b \text{pop}_b (1 + v_s \cdot \text{visitors}_b) K(d_{ab})}{\text{same for all of Ajman}} \qquad K(d) = e^{-d/2\,\text{km}}")
    st.latex(r"\text{Saturation}_{a,s} = \ell_s\, g(\text{ratio}_{a,s}) + (1-\ell_s)\cdot 0.2\, g(LQ_{a,s}) \qquad g(r) = \text{clip}\left(\tfrac{r - 0.75}{2.25}, 0, 1\right)")
    st.markdown("Competitors per customer in the catchment, relative to Ajman. $\\ell_s$ = how local a sector's customers are "
                "(F&B 1.0, retail 1.0, tourism 0.8 … manufacturing 0.1); $v_s$ = how much visitors add to demand. "
                "Population and visitor figures are **estimates** in `data/geo/ajman_areas.csv`; replace with official "
                "Ajman Statistics Center data.")

    st.subheader("How well it works (test cohort: licensed 2022+)")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("C-index", f"{M['c_index_score']:.3f}", help="0.5 = random ranking, 1 = perfect")
    q2.metric("Brier @ 3 yr", f"{M['brier_36m']:.3f}", f"{M['brier_36m'] - M['brier_36m_null']:+.3f} vs no-model baseline",
              delta_color="inverse")
    q3.metric("AUC @ 3 yr (score)", f"{M.get('auc36_score', float('nan')):.3f}")
    q4.metric("AUC @ 3 yr (XGBoost)", f"{M.get('auc36_xgboost', float('nan')):.3f}")
    st.caption("The explainable Cox-based score beats a black-box XGBoost baseline. Time-based split: trained on businesses "
               f"licensed up to mid-2020, tested on 2022+ cohorts. Putting saturation inside the model beat the earlier "
               f"multiplicative penalty (C-index {M['c_index_score']:.3f} vs {M['c_index_penalty_variant']:.3f}).")
    cal_img = REPORTS / "figures" / "calibration_36m.png"
    if cal_img.exists():
        st.image(str(cal_img), width=380)
    if M.get("ph_violations_at_0.01"):
        st.caption("Proportional-hazards check flags: " + ", ".join(M["ph_violations_at_0.01"]) +
                   " (with ~9,000 businesses even small departures are significant; effects are stable enough for ranking).")
    st.subheader("Fairness & limits")
    st.markdown("- No personal attributes (nationality, gender, age) are used.\n"
                "- Decision support only; not a licensing gate.\n"
                "- Trained on **synthetic data**; retrain on real DED license records before any real use.")

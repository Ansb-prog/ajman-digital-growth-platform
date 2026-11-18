"""The three program pillars as Streamlit sections (imported by streamlit_app.py)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from app import theme
from src.growth.program import HAZARD_REDUCTION, MARKETING_RELEVANCE, PACKAGE, district_opportunity, program_cost
from src.viz.risk_map import build_growth_map

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
INK, MUTED, GREEN, RED, BLUE = "#1f2937", "#6b7280", "#2e8b57", "#c0392b", "#2471a3"


@st.cache_data
def program_tables():
    return {
        "effects": pd.read_csv(PROC / "program_marketing_effects.csv"),
        "scen": pd.read_csv(PROC / "program_scenarios.csv"),
        "creators": pd.read_csv(PROC / "program_creators.csv"),
        "ew": pd.read_csv(PROC / "program_early_warning.csv"),
        "matches": pd.read_csv(PROC / "program_matches.csv"),
        "priority": pd.read_csv(PROC / "program_priority_districts.csv"),
        "supply": json.loads((REPORTS / "program_supply.json").read_text()),
        "afz_type": pd.read_csv(PROC / "forecast_afz_by_type.csv"),
        "afz_monthly": pd.read_csv(ROOT / "data" / "real" / "afz_monthly.csv", parse_dates=["month"]),
        "districts_fc": pd.read_csv(PROC / "forecast_districts.csv"),
    }


# ------------------------------------------------------------------ pillar 1
def registration_opportunity(biz: dict, bundle, r: dict) -> None:
    """Rendered inside the advisor tab, below the applicant's own score."""
    P = program_tables()
    theme.section("All options for this license, ranked", "REAL DATA", "MODEL")
    st.caption("Opportunity = 50% survival score in that district + 30% forecast growth 2025–2028 (vs Ajman) + "
               "20% room in the market (1 − saturation). Only districts where this sector already operates "
               "(≥1% of its licenses) are shown, as a zoning proxy.")
    opp = district_opportunity({k: v for k, v in biz.items() if k != "area"}, bundle)
    prio = set(P["priority"]["district"])
    c1, c2 = st.columns([1.1, 1])
    incentive = c1.toggle("Apply DED priority-district incentive (−20% registration fee)", value=True,
                          help="Proposed policy: fee discount in districts growing faster than Ajman with low saturation.")
    opp["incentive"] = opp["district"].isin(prio)
    if incentive:
        opp["opportunity"] = opp["opportunity"] + np.where(opp["incentive"], 3.0, 0.0)  # visible nudge, not a mandate
        opp = opp.sort_values("opportunity", ascending=False)
    hide_remote = c2.toggle("Hide Masfout/Manama (remote licensing hubs)", value=True)
    view = opp[~opp["remote_hub"]] if hide_remote else opp
    cur = opp[opp["district"] == biz["area"]]
    best = view.iloc[0]
    if len(cur) and best["district"] != biz["area"]:
        cu = cur.iloc[0]
        st.success(f"**{best['district']}** ({best['parent']}) scores **{best['opportunity']:.0f}** vs "
                   f"**{cu['opportunity']:.0f}** in {biz['area']}: survival {best['score']:.0f} vs {cu['score']:.0f}, "
                   f"forecast growth {best['growth_pct']:+.0f}% vs {cu['growth_pct']:+.0f}%, saturation "
                   f"{best['saturation']:.2f} vs {cu['saturation']:.2f}."
                   + (" Eligible for the priority-district incentive." if best["incentive"] and incentive else ""))
    t = view.head(8)[["district", "parent", "opportunity", "score", "growth_pct", "saturation", "outlook", "incentive"]].copy()
    t.columns = ["District", "Area", "Opportunity", "Survival score", "Growth 25→28 %", "Saturation", "Outlook", "Incentive"]
    st.dataframe(t.round(1), hide_index=True, width="stretch")
    with st.expander("Map: forecast growth by district (green = growing faster than Ajman)"):
        st_folium(build_growth_map(P["districts_fc"]), height=430, use_container_width=True, returned_objects=[],
                  key="adv_map")
    st.caption("Priority districts (proposed incentive list): " + ", ".join(P["priority"]["district"].head(10)))


# ------------------------------------------------------------------ pillar 2
def creator_hub() -> None:
    P = program_tables()
    S = P["supply"]
    st.subheader("② Creator Hub: make Ajman the base for social media marketers")
    st.markdown(theme.badge("REAL DATA") + " freelance license counts (AFZ) " + theme.badge("ESTIMATE") +
                " media/marketing share " + theme.badge("ASSUMPTION") + " package uplift", unsafe_allow_html=True)
    c = st.columns(4)
    c[0].metric("Freelance licenses, last 12 months", f"{S['freelancers_last_12m']:,.0f}", help="Ajman Free Zone (real)")
    c[1].metric("Forecast, next 12 months", f"{S['freelancers_next_12m_baseline']:,.0f}",
                f"{S['freelancers_next_12m_baseline'] / S['freelancers_last_12m'] - 1:+.0%}")
    c[2].metric("of which media/marketing (est.)", f"{S['media_creators_next_12m_baseline']:,.0f}",
                help="Assumes 35% of freelancers do media/marketing; ask FZA for the real split (AFZ + Ajman Media City).")
    c[3].metric("E-commerce licenses, next 12 months", f"{S['ecommerce_next_12m']:,.0f}", help="AFZ forecast (real data)")

    m = P["afz_monthly"]
    m = m[~m["partial_month"]]
    fig = px.bar(m, x="month", y="License: Free Lancer", labels={"License: Free Lancer": "New freelance licenses", "month": ""})
    fig.update_traces(marker_color=BLUE)
    fig.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Incentive scenario")
    a, b = st.columns([1, 1.4])
    with a:
        uplift = st.slider("Extra freelance registrations from the Creator package (%)", 0, 60, 25, 5,
                           help="Assumption to test: one-stop license + advertiser-permit help, mainland access, "
                                "renewal discount for serving Ajman SMEs.")
        share = st.slider("Share of freelancers doing media/marketing (%)", 10, 70, 35, 5)
        clients = st.slider("SME clients one creator can serve", 3, 15, 8)
    base = S["freelancers_next_12m_baseline"]
    with_pkg = base * (1 + uplift / 100)
    creators = with_pkg * share / 100
    sc = P["scen"].drop_duplicates("scenario")[["scenario", "matched_per_year"]]
    sc["creators_needed"] = sc["matched_per_year"] / clients
    with b:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=sc["scenario"], y=sc["creators_needed"], name="Creators needed", marker_color="#aab7c4"))
        fig.add_hline(y=creators, line_dash="dash", line_color=GREEN,
                      annotation_text=f"Media creators available: {creators:,.0f}")
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="Creators",
                          title="Can Ajman's creators serve the program?")
        st.plotly_chart(fig, width="stretch")
    st.caption(f"Freelance licenses next 12 months: {base:,.0f} baseline → {with_pkg:,.0f} with the package "
               f"(+{with_pkg - base:,.0f}).")

    st.markdown("#### Levers DED and FZA control")
    st.markdown(
        "- **One-stop Creator package (FZA / Ajman Media City):** freelance license + visa + help with the federal UAE Media "
        "Council advertiser permit, which every paid creator now needs (from 1 Feb 2026).\n"
        "- **Mainland access (DED + FZA):** let Ajman free zone creators legally serve Ajman mainland businesses.\n"
        "- **Promotion permits (DED):** optional 'licensed creator' field on discount/offer permits (1,148 issued in 2025) "
        "→ fast-track or fee cut when an Ajman-licensed creator runs the campaign.\n"
        "- **Renewal incentives:** lower renewal fees for creators who served ≥N Ajman SMEs, and for SMEs that used one.\n"
        "- **Directory:** only Ajman-licensed creators are listed in the Growth Matchmaker.")
    cost = pd.DataFrame({"Route": ["Ajman Free Zone freelance license", "Dubai GoFreelance (TECOM)"],
                         "Indicative annual license fee (AED)": ["≈ 6,000 – 7,500", "≈ 7,500"],
                         "Source": ["dubairealestate.net (2026 overview)", "DDA published fee via stashaway.ae (2026)"]})
    st.dataframe(cost, hide_index=True, width="stretch")
    st.caption("Fees from public third-party summaries; confirm current official fees with FZA before pitching.")


# ------------------------------------------------------------------ pillar 3
def matchmaker() -> None:
    P = program_tables()
    E, SC = P["effects"], P["scen"]
    st.subheader("③ Growth Matchmaker: connect struggling businesses with Ajman creators")

    theme.section("Estimated effect of social media marketing, by sector", "ASSUMPTION", "MODEL")
    st.caption(f"Effect of a 6-month creator package ({PACKAGE['posts_per_month']} posts/month, ~3% engagement, website, "
               f"+{PACKAGE['extra_positive_reviews']} positive reviews) on a **weak-online** business: closure hazard −"
               f"{HAZARD_REDUCTION['mid']:.0%} (range {HAZARD_REDUCTION['low']:.0%}–{HAZARD_REDUCTION['high']:.0%}) × sector relevance. "
               "This size is an assumption guided by published evidence (social media advertising reduced small-business "
               "closures during lockdowns) and is what the pilot measures.")
    e = E.sort_values("risk_reduction_pp")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=e["sector"], x=e["risk_reduction_pp"], orientation="h", marker_color=[
        GREEN if r >= 0.9 else ("#7dbb5a" if r >= 0.6 else "#aab7c4") for r in e["relevance"]],
        error_x=dict(type="data", symmetric=False, array=e["risk_reduction_pp_high"] - e["risk_reduction_pp"],
                     arrayminus=e["risk_reduction_pp"] - e["risk_reduction_pp_low"], color=MUTED),
        hovertemplate="%{y}: −%{x:.1f} pp<extra></extra>", name="Risk reduction"))
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="Fewer closures per 100 matched businesses (12 months)")
    st.plotly_chart(fig, width="stretch")
    t = E[["sector", "relevance", "p_close_12m_now", "risk_reduction_pp", "relative_reduction_pct", "score_uplift",
           "weak_online_businesses"]].copy()
    t.columns = ["Sector", "Marketing relevance", "12-m closure risk now", "Fewer closures per 100", "Relative cut %",
                 "Score uplift", "Weak-online businesses (est.)"]
    st.dataframe(t.round(2), hide_index=True, width="stretch")
    st.caption("Biggest gains: customer-facing sectors with high baseline risk (F&B, retail, tourism, real estate) – the same "
               "sectors forecast to grow fastest. B2B sectors (construction, manufacturing) gain little from social media.")

    theme.section("Program impact on the forecast", "ASSUMPTION", "MODEL")
    name = st.radio("Scenario", SC["scenario"].unique().tolist(), horizontal=True, index=1)
    s = SC[SC["scenario"] == name]
    k1, k2, k3 = st.columns(3)
    k1.metric("Businesses matched per year", f"{s['matched_per_year'].iloc[0]:,.0f}")
    k2.metric("Closures avoided by 2028", f"{s['exits_avoided'].sum():,.0f}",
              f"range {s['exits_avoided_low'].sum():,.0f}–{s['exits_avoided_high'].sum():,.0f}", delta_color="off")
    k3.metric("Extra active licenses, end-2028", f"{s['cum_extra_active'].iloc[-1]:,.0f}")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=s["year"], y=s["exits_avoided"], name="Closures avoided", marker_color=GREEN,
                         error_y=dict(type="data", symmetric=False, array=s["exits_avoided_high"] - s["exits_avoided"],
                                      arrayminus=s["exits_avoided"] - s["exits_avoided_low"])))
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Closures avoided per year")
    st.plotly_chart(fig, width="stretch")
    st.caption("Baseline forecast exits: " + ", ".join(f"{int(y)}: {v:,.0f}" for y, v in zip(s["year"], s["baseline_exits"]))
               + ". The program's main value is concentrated: it saves businesses in the growth sectors and districts, "
                 "and creates demand for Ajman-licensed creators.")

    theme.section("Early-warning list and matches", "SIMULATED")
    st.caption("Active businesses in customer-facing sectors with high 12-month closure risk and weak online presence, "
               "each matched to a creator by sector specialty, distance, budget and rating. **Businesses and creators are "
               "simulated**; with DED's register this list is generated from real licenses, and the research agent "
               "audits each business's real online presence.")
    M = P["matches"]
    f1, f2 = st.columns(2)
    sec = f1.multiselect("Sector", sorted(M["sector"].unique()), default=[])
    dist = f2.multiselect("District", sorted(M["area"].unique()), default=[])
    mask = pd.Series(True, index=M.index)
    if sec:
        mask &= M["sector"].isin(sec)
    if dist:
        mask &= M["area"].isin(dist)
    view = M[mask]
    show = view[["license_id", "sector", "area", "score", "p_close_12m", "p_close_12m_after", "creator_id",
                 "creator_type", "creator_specialty", "creator_district", "distance_km", "monthly_fee_aed"]].head(25).copy()
    show["p_close_12m"] = (100 * show["p_close_12m"]).round(0).astype(int).astype(str) + "%"
    show["p_close_12m_after"] = (100 * show["p_close_12m_after"]).round(0).astype(int).astype(str) + "%"
    show.columns = ["Business", "Sector", "District", "Score", "Closure risk", "After match", "Creator", "Type",
                    "Specialty", "Creator based in", "Distance km", "Fee AED/month"]
    st.dataframe(show, hide_index=True, width="stretch")
    if len(view):
        pick = st.selectbox("Business card", view["license_id"].head(25).tolist())
        row = view[view["license_id"] == pick].iloc[0]
        a, b, c = st.columns(3)
        a.metric("Survival score", f"{row['score']:.0f}", f"{row['score_after'] - row['score']:+.0f} after match")
        b.metric("12-month closure risk", f"{row['p_close_12m']:.0%}", f"{100 * (row['p_close_12m_after'] - row['p_close_12m']):+.1f} pp",
                 delta_color="inverse")
        c.metric("Now posting", f"{row['posts_per_month']:.0f}/month", f"{PACKAGE['posts_per_month']}/month with creator",
                 delta_color="off")
        st.caption(f"{row['sector']} in {row['area']}, open {row['duration_months']:.0f} months · Google "
                   f"{row['google_rating'] if pd.notna(row['google_rating']) else '–'}★ ({row['google_review_count']:.0f} reviews) · "
                   f"matched with {row['creator_type'].lower()} {row['creator_id']} ({row['creator_specialty']}, "
                   f"{row['creator_district']}, {row['distance_km']} km, AED {row['monthly_fee_aed']:,}/month).")

    theme.section("Cost per outcome", "ASSUMPTION")
    c1, c2, c3 = st.columns(3)
    v = c1.slider("Voucher (AED per business per month)", 500, 3000, 1250, 250,
                  help="Public share of a creator's fee, e.g. 50% of AED 2,500/month.")
    mo = c2.slider("Voucher months", 3, 12, 6)
    adm = c3.slider("Admin overhead (%)", 0, 30, 15, 5) / 100
    cost = program_cost(SC, v, mo, adm)
    show = cost[["scenario", "businesses_matched_total", "cost_aed_total", "closures_avoided",
                 "cost_per_closure_avoided", "cost_per_closure_avoided_best", "cost_per_closure_avoided_worst",
                 "cost_per_job_kept"]].copy()
    show.columns = ["Scenario (2026–2028)", "Businesses matched", "Total cost AED", "Closures avoided",
                    "AED per closure avoided", "best case", "worst case", "AED per job kept (≈3 jobs/business)"]
    st.dataframe(show.round(0), hide_index=True, width="stretch")
    st.caption("Counts only avoided closures. Not counted: extra revenue of matched businesses, new creator licenses, "
               "and renewal fees kept. These are the numbers the pilot should measure before scaling.")

    theme.section("Pilot to propose")
    st.markdown("- **200 businesses** in F&B, retail and services, in priority districts (Al Zahya, Al Amerah, Al Aaliah, Jurf 2, Al Helio).\n"
                "- Half matched with an Ajman-licensed creator + voucher; half as comparison group.\n"
                "- Measure after 12 months: renewal rate, reviews, promotion permits → replaces the assumed effect with Ajman's real one.\n"
                "- Data request: anonymized license history (DED) and freelance/media licenses by activity (AFZ + Ajman Media City).")

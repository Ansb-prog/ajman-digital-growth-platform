"""Ajman Digital Growth program: the engine behind the three pillars.

Pillar 1  Registration Advisor   opportunity score per district for an applicant (survival + growth + saturation)
Pillar 2  Creator Hub            freelancer / agency supply: baseline forecast, incentive scenarios, capacity gap
Pillar 3  Growth Matchmaker      early-warning list, creator matching, estimated effect of social media marketing

WHAT IS REAL vs ASSUMED
- Real: district growth forecasts, saturation, sector stock sizes, freelancer registrations (AFZ), exits (DED).
- Model: survival model (calibrated simulation); the size of the digital-presence effect on survival is an
  ASSUMPTION of the simulation (no public Ajman data links online activity to closures). Every effect below is
  therefore shown with a sensitivity range (0.5x - 1.5x) and is meant to be measured in the proposed pilot.
- Assumed: creator profiles, prices, capacity; sector relevance of social media marketing (below).

Run:  python -m src.growth.program
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data.load import ROOT, load_businesses
from src.features.saturation import kernel_matrix, load_areas
from src.models.cox import survival_at
from src.models.score import DEFAULT_BUSINESS, Bundle, load_bundle, score_frame

PROC = ROOT / "data" / "processed"
REAL = ROOT / "data" / "real"
REPORTS = ROOT / "reports"

# How much of the model's digital effect applies, by how customer-facing the sector is (ASSUMPTION).
MARKETING_RELEVANCE = {
    "F&B": 1.0, "retail": 1.0, "tourism": 1.0, "personal services": 1.0, "real estate": 0.9,
    "tech": 0.7, "professional services": 0.7, "auto services": 0.6,
    "trading": 0.4, "construction": 0.3, "manufacturing": 0.2,
}

# What a matched creator can realistically change within ~6 months (the "marketing package").
PACKAGE = {"posts_per_month": 12, "days_since_last_post": 3, "engagement_rate": 0.03,
           "instagram_followers": 1500, "has_website": True, "extra_positive_reviews": 20}

LOW_DIGITAL = 0.30  # Digital Presence Index below this = "weak online"

# EFFECT SIZE (transparent assumption, not estimated from Ajman data):
# a matched, customer-facing, weak-online business cuts its closure hazard by 20% (range 10-30%), times the
# sector's relevance. Published evidence is directional only (e.g. social media advertising reduced small-business
# closures during COVID lockdowns: Lee et al. 2025, Journal of Interactive Marketing) and gives no Ajman number, so
# the pilot is designed to measure it. The survival model's own implied effect is much larger (it inherits the
# simulation's assumption) and is reported only as an upper bound.
HAZARD_REDUCTION = {"low": 0.10, "mid": 0.20, "high": 0.30}


# ------------------------------------------------------------------ pillar 1
# At registration a business has no track record online, so the REGISTRATION score uses only what is known
# (sector, district, capital, size, legal form, premises) and sets marketing/reputation to the Ajman median.
NEUTRAL_MARKETING = {"posts_per_month": 4, "engagement_rate": 0.04, "instagram_followers": 590,
                     "days_since_last_post": 10, "has_website": False, "google_review_count": 0, "google_rating": float("nan")}
MAX_SUGGESTED_SATURATION = 0.30  # do not steer applicants into a district beyond this


def registration_view(business: dict) -> dict:
    """The business with marketing/reputation fields replaced by neutral values."""
    return {**business, **NEUTRAL_MARKETING}


def headroom(sat_row: pd.Series, nearby: float, target: float = MAX_SUGGESTED_SATURATION) -> float:
    """Approx. extra same-sector licenses (within the ~2 km catchment) before saturation reaches `target`.
    Uses Saturation ~= local_share x clip((ratio - 0.75)/2.25) with ratio proportional to competitors nearby."""
    l = max(float(sat_row["local_share"]), 0.1)
    r_star = 0.75 + 2.25 * min(target / l, 1.0)
    ratio = max(float(sat_row["ratio"]), 1e-6)
    return max(0.0, nearby * (r_star / ratio - 1))
def district_opportunity(business: dict, bundle: Bundle | None = None, min_share: float = 0.01) -> pd.DataFrame:
    """Score the applicant's business in every district where its sector already operates, and rank by opportunity.

    opportunity = 0.5 x survival score (0-100) + 0.3 x growth index (0-100) + 0.2 x (1 - saturation) x 100
    growth index = district's forecast growth 2025-2028 relative to Ajman, mapped to 0-100 (50 = Ajman average).
    """
    b = bundle or load_bundle()
    sat = b.saturation_table
    sec = business.get("sector", DEFAULT_BUSINESS["sector"])
    s = sat[sat["sector"] == sec]
    tot = max(s["n"].sum(), 1)
    areas = s.loc[s["n"] / tot >= min_share, "area"].tolist()
    rows = pd.DataFrame([{**DEFAULT_BUSINESS, **registration_view(business), "area": a} for a in areas])
    sc = score_frame(b, rows)
    fd = pd.read_csv(PROC / "forecast_districts.csv").set_index("district")
    out = pd.DataFrame({"district": areas, "score": sc["score"].values, "saturation": sc["saturation"].values})
    out["parent"] = out["district"].map(fd["parent"])
    out["growth_pct"] = out["district"].map(fd["growth_2025_2028_pct"])
    out["outlook"] = out["district"].map(fd["outlook"]).fillna("n/a")
    out["sector_licenses"] = out["district"].map(s.set_index("area")["n"])
    vs = out["district"].map(fd["vs_ajman_pp"]).fillna(0)
    out["growth_index"] = (50 + vs.clip(-50, 50)).clip(0, 100)
    out["opportunity"] = 0.5 * out["score"] + 0.3 * out["growth_index"] + 0.2 * (1 - out["saturation"]) * 100
    # remote licensing hubs: growth there often reflects registrations of businesses operating elsewhere
    out["remote_hub"] = out["parent"].isin(["Masfout", "Manama"])
    comp = competition(sec)
    out = out.merge(comp, on="district", how="left")
    srow = s.set_index("area")
    out["headroom"] = [headroom(srow.loc[d], nb) if d in srow.index else 0.0
                       for d, nb in zip(out["district"], out["competitors_nearby"])]
    return out.sort_values("opportunity", ascending=False).reset_index(drop=True)


def competition(sector: str) -> pd.DataFrame:
    """REAL competition for a sector in every district: own-district licenses, competitors within ~2 km
    (distance-weighted, same kernel as saturation) and competitors per 10k residents (estimated)."""
    areas = load_areas().reset_index(drop=True)
    from src.features.saturation import real_counts
    rc = real_counts()
    n = rc[rc["sector"] == sector].set_index("area")["n"].reindex(areas["area"]).fillna(0).values
    K = kernel_matrix(areas)
    near = K @ n
    pop = areas["population_est"].clip(lower=1000).values
    return pd.DataFrame({"district": areas["area"], "competitors": n.astype(int), "competitors_nearby": near.round(0),
                         "competitors_per_10k": (1e4 * near / (K @ pop)).round(1)})


def suggestions(business: dict, bundle: Bundle | None = None, k: int = 3) -> tuple[pd.DataFrame, pd.Series | None]:
    """Districts with LESS competition AND better metrics than the applicant's chosen district."""
    opp = district_opportunity({kk: v for kk, v in business.items() if kk != "area"}, bundle)
    return pick_suggestions(opp, business.get("area"), k)


def pick_suggestions(opp: pd.DataFrame, current: str | None, k: int = 3) -> tuple[pd.DataFrame, pd.Series | None]:
    cur_all = opp[opp["district"] == current]
    # never steer into crowded markets: must stay below the saturation cap with room for >= 10 more licenses
    opp = opp[~opp["remote_hub"] & ((opp["district"] == current) |
                                    ((opp["saturation"] < MAX_SUGGESTED_SATURATION) & (opp["headroom"] >= 10)))]
    cur = cur_all
    cur = cur.iloc[0] if len(cur) else None
    cand = opp.copy()
    if cur is not None:
        cand = cand[(cand["district"] != cur["district"])
                    & (cand["competitors_per_10k"] < cur["competitors_per_10k"])
                    & (cand["score"] >= cur["score"] - 1) & (cand["growth_pct"] >= cur["growth_pct"])]
        if len(cand) < k:  # relax: just less competition and better opportunity
            extra = opp[(opp["district"] != cur["district"]) & (opp["opportunity"] > cur["opportunity"])
                        & ~opp["district"].isin(cand["district"])]
            cand = pd.concat([cand, extra])
    return cand.sort_values("opportunity", ascending=False).head(k).reset_index(drop=True), cur


def priority_districts() -> pd.DataFrame:
    """Districts DED could prioritise with a registration incentive: growing faster than Ajman and not saturated."""
    fd = pd.read_csv(PROC / "forecast_districts.csv")
    area = pd.read_csv(PROC / "area_stats.csv").set_index("area")
    fd["avg_saturation"] = fd["district"].map(area["avg_saturation"])
    p = fd[(fd["outlook"] == "Hotspot") & (fd["avg_saturation"] < 0.2) & (fd["city"] == "Ajman")]
    return p.sort_values("growth_2025_2028_pct", ascending=False)


# ------------------------------------------------------------------ pillar 3 effects
def apply_package(df: pd.DataFrame) -> pd.DataFrame:
    """Digital signals after a creator engagement (only ever improves a signal)."""
    d = df.copy()
    for k in ("posts_per_month", "engagement_rate", "instagram_followers"):
        d[k] = np.maximum(d[k].fillna(0), PACKAGE[k])
    d["days_since_last_post"] = np.minimum(d["days_since_last_post"].fillna(180), PACKAGE["days_since_last_post"])
    d["has_website"] = True
    n0 = d["google_review_count"].fillna(0)
    r0 = d["google_rating"].fillna(4.3).where(n0 > 0, 4.3)
    add = PACKAGE["extra_positive_reviews"]
    d["google_review_count"] = n0 + add
    d["google_rating"] = (r0 * n0 + 4.6 * add) / (n0 + add)  # new reviews average 4.6 stars
    return d


def _p_close(b: Bundle, df: pd.DataFrame, horizon: int = 12, effect_scale: float = 1.0,
             base_df: pd.DataFrame | None = None) -> np.ndarray:
    """P(close within `horizon` months) given current age. effect_scale shrinks/grows the digital+reputation change."""
    x = b.features.transform(df)
    if base_df is not None and effect_scale != 1.0:
        x0 = b.features.transform(base_df)
        for c in ("digital_presence", "reputation"):
            x[c] = x0[c] + effect_scale * (x[c] - x0[c])
    age = df["duration_months"].values
    s_now = survival_at(b.cox, x, age)
    return 1 - survival_at(b.cox, x, age + horizon) / np.clip(s_now, 1e-9, None)


def marketing_effects(b: Bundle | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sector-level effect of the marketing package on weak-online businesses, scaled to Ajman's real stock."""
    b = b or load_bundle()
    df = load_businesses()
    act = df[df["event"] == 0].reset_index(drop=True)
    d0 = score_frame(b, act)["digital_presence"].values
    weak = act[d0 < LOW_DIGITAL].reset_index(drop=True)
    after = apply_package(weak)
    stock = pd.read_csv(PROC / "stock_profile_2025.csv")
    real_active = float(pd.read_csv(REAL / "annual.csv").set_index("year").loc[2025, "active"])
    sector_stock = stock.groupby("sector")["weight"].sum()
    sector_stock = sector_stock / sector_stock.sum() * real_active
    rows = []
    for s, g in weak.groupby("sector"):
        idx = g.index
        rel = MARKETING_RELEVANCE[s]
        p0 = _p_close(b, g)
        p_model = _p_close(b, after.loc[idx], effect_scale=rel, base_df=g)
        res = {k: float(np.mean(p0 - (1 - (1 - p0) ** (1 - v * rel)))) for k, v in HAZARD_REDUCTION.items()}
        s36 = score_frame(b, g)["score"].values / 100
        share_weak = len(g) / max((act["sector"] == s).sum(), 1)
        rows.append({"sector": s, "relevance": rel, "weak_online_share": share_weak,
                     "weak_online_businesses": share_weak * sector_stock.get(s, 0),
                     "p_close_12m_now": float(p0.mean()),
                     "risk_reduction_pp": 100 * res["mid"], "risk_reduction_pp_low": 100 * res["low"],
                     "risk_reduction_pp_high": 100 * res["high"],
                     "relative_reduction_pct": 100 * res["mid"] / max(float(p0.mean()), 1e-9),
                     "model_upper_bound_relative_pct": 100 * float(np.mean(p0 - p_model)) / max(float(p0.mean()), 1e-9),
                     "score_now": 100 * float(s36.mean()),
                     "score_uplift": 100 * float(np.mean(s36 ** (1 - HAZARD_REDUCTION["mid"] * rel) - s36)),
                     "closures_avoided_per_100_matched": 100 * res["mid"]})
    eff = pd.DataFrame(rows).sort_values("closures_avoided_per_100_matched", ascending=False)

    # program scenarios -> effect on the 2026-2028 forecast
    fc = pd.read_csv(PROC / "forecast_annual.csv")
    target = eff[eff["relevance"] >= 0.7]  # program targets customer-facing sectors
    pool = target["weak_online_businesses"].sum()
    wavg = lambda col: float(np.average(target[col], weights=target["weak_online_businesses"]))  # noqa: E731
    scen = []
    for name, matched in (("Pilot (200 businesses)", 200), ("Year-1 program (1,000)", 1000),
                          ("Scale-up (25% of weak-online, target sectors)", 0.25 * pool)):
        for y_i, y in enumerate(fc["year"]):
            cum = matched * (1 + y_i)  # same number matched each year, effect lasts while the business survives
            avoided = {k: cum * wavg(c) / 100 for k, c in (("mid", "risk_reduction_pp"), ("low", "risk_reduction_pp_low"),
                                                          ("high", "risk_reduction_pp_high"))}
            scen.append({"scenario": name, "matched_per_year": matched, "year": int(y),
                         "exits_avoided": avoided["mid"], "exits_avoided_low": avoided["low"],
                         "exits_avoided_high": avoided["high"], "baseline_exits": float(fc.loc[y_i, "exits"]),
                         "baseline_active": float(fc.loc[y_i, "active"])})
    scen = pd.DataFrame(scen)
    scen["cum_extra_active"] = scen.groupby("scenario")["exits_avoided"].cumsum()
    scen["active_with_program"] = scen["baseline_active"] + scen["cum_extra_active"]
    return eff, scen


def program_cost(scen: pd.DataFrame, voucher_aed_per_month: float = 1250, months: int = 6,
                 admin_share: float = 0.15, jobs_per_business: float = 3.0) -> pd.DataFrame:
    """Cost per outcome. Voucher = public share of a creator's fee (e.g. 50% of AED 2,500/month), for `months`."""
    out = []
    for name, g in scen.groupby("scenario", sort=False):
        n = float(g["matched_per_year"].iloc[0]) * len(g)
        cost = n * voucher_aed_per_month * months * (1 + admin_share)
        saved = float(g["exits_avoided"].sum())
        lo, hi = float(g["exits_avoided_low"].sum()), float(g["exits_avoided_high"].sum())
        out.append({"scenario": name, "businesses_matched_total": n, "cost_aed_total": cost,
                    "closures_avoided": saved, "closures_avoided_low": lo, "closures_avoided_high": hi,
                    "cost_per_closure_avoided": cost / max(saved, 1e-9),
                    "cost_per_closure_avoided_best": cost / max(hi, 1e-9),
                    "cost_per_closure_avoided_worst": cost / max(lo, 1e-9),
                    "jobs_kept_est": saved * jobs_per_business,
                    "cost_per_job_kept": cost / max(saved * jobs_per_business, 1e-9)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ pillar 2
def creator_supply(scen: pd.DataFrame) -> dict:
    """Baseline freelancer forecast (real AFZ data) and capacity needed by the matching program."""
    af = pd.read_csv(PROC / "forecast_afz_by_type.csv").set_index("license_type")
    fl_last, fl_next = float(af.loc["Free Lancer", "last_12m"]), float(af.loc["Free Lancer", "next_12m"])
    ecom_next = float(af.loc["E-Commerce", "next_12m"])
    out = {"freelancers_last_12m": fl_last, "freelancers_next_12m_baseline": fl_next,
           "ecommerce_next_12m": ecom_next,
           "assumed_share_media_marketing": 0.35, "assumed_clients_per_creator": 8}
    out["media_creators_next_12m_baseline"] = fl_next * out["assumed_share_media_marketing"]
    for name, g in scen.groupby("scenario"):
        need = float(g["matched_per_year"].iloc[0]) / out["assumed_clients_per_creator"]
        out[f"creators_needed::{name}"] = need
    return out


def creator_pool(n: int = 220, seed: int = 7) -> pd.DataFrame:
    """SIMULATED marketers (freelancers + agencies) for the matching demo. Size anchored to ~35% of AFZ freelancers."""
    rng = np.random.default_rng(seed)
    areas = load_areas()
    areas = areas[areas["city"] == "Ajman"]
    w = areas["population_est"].clip(lower=1000).values
    home = rng.choice(areas["area"].values, size=n, p=w / w.sum())
    specs = list(MARKETING_RELEVANCE)
    pspec = np.array([MARKETING_RELEVANCE[s] for s in specs])
    kind = np.where(rng.random(n) < 0.8, "Freelancer", "Agency")
    first = rng.choice(specs, n, p=pspec / pspec.sum())
    second = rng.choice(specs, n, p=pspec / pspec.sum())
    price = np.where(kind == "Agency", rng.normal(6500, 1500, n), rng.normal(2500, 800, n)).clip(800)
    return pd.DataFrame({
        "creator_id": [f"CR-{i:03d}" for i in range(1, n + 1)], "type": kind, "home_district": home,
        "specialty_1": first, "specialty_2": second,
        "monthly_fee_aed": (price // 100 * 100).astype(int),
        "rating": rng.normal(4.5, 0.3, n).clip(3.5, 5.0).round(1),
        "capacity": np.where(kind == "Agency", rng.integers(10, 25, n), rng.integers(3, 9, n)),
        "ajman_licensed": True, "is_simulated": True,
    })


def early_warning(b: Bundle | None = None, top: int = 400) -> pd.DataFrame:
    """Active businesses with high 12-month closure risk AND weak online presence (simulated register)."""
    b = b or load_bundle()
    df = load_businesses()
    act = df[df["event"] == 0].reset_index(drop=True)
    sc = score_frame(b, act)
    act = pd.concat([act, sc[["score", "digital_presence"]]], axis=1)
    act["p_close_12m"] = _p_close(b, act)
    act["relevance"] = act["sector"].map(MARKETING_RELEVANCE)
    cand = act[(act["digital_presence"] < LOW_DIGITAL) & (act["relevance"] >= 0.7)].copy()
    rr = HAZARD_REDUCTION["mid"] * cand["relevance"].values
    cand["p_close_12m_after"] = 1 - (1 - cand["p_close_12m"]) ** (1 - rr)
    cand["score_after"] = 100 * (cand["score"] / 100) ** (1 - rr)
    cand["risk_drop_pp"] = 100 * (cand["p_close_12m"] - cand["p_close_12m_after"])
    cand["priority"] = cand["p_close_12m"] * cand["risk_drop_pp"]
    cols = ["license_id", "sector", "area", "duration_months", "capital_aed", "employees", "score", "digital_presence",
            "p_close_12m", "p_close_12m_after", "risk_drop_pp", "score_after", "posts_per_month",
            "google_review_count", "google_rating", "priority"]
    return cand.sort_values("priority", ascending=False)[cols].head(top).reset_index(drop=True)


def match(ew: pd.DataFrame, creators: pd.DataFrame, budget_aed: int = 3000) -> pd.DataFrame:
    """Greedy capacity-constrained matching: specialty fit, distance, budget fit and rating."""
    areas = load_areas().set_index("area")
    cap = creators.set_index("creator_id")["capacity"].to_dict()
    C = creators.set_index("creator_id")
    lat_c = C["home_district"].map(areas["lat"]).values
    lon_c = C["home_district"].map(areas["lon"]).values
    rows = []
    for _, bz in ew.iterrows():
        la, lo = areas.loc[bz["area"], "lat"], areas.loc[bz["area"], "lon"]
        dist = np.hypot((lat_c - la) * 111, (lon_c - lo) * 101)
        spec = np.where(C["specialty_1"].values == bz["sector"], 1.0,
                        np.where(C["specialty_2"].values == bz["sector"], 0.6, 0.0))
        budget = np.clip(1 - np.maximum(C["monthly_fee_aed"].values - budget_aed, 0) / budget_aed, 0, 1)
        fit = 0.45 * spec + 0.25 * np.exp(-dist / 6) + 0.15 * budget + 0.15 * (C["rating"].values - 3.5) / 1.5
        avail = np.array([cap[c] > 0 for c in C.index])
        fit = np.where(avail, fit, -1)
        j = int(np.argmax(fit))
        if fit[j] < 0:
            continue
        cid = C.index[j]
        cap[cid] -= 1
        rows.append({**bz.to_dict(), "creator_id": cid, "creator_type": C.loc[cid, "type"],
                     "creator_district": C.loc[cid, "home_district"], "creator_specialty": C.loc[cid, "specialty_1"],
                     "monthly_fee_aed": int(C.loc[cid, "monthly_fee_aed"]), "creator_rating": C.loc[cid, "rating"],
                     "distance_km": round(float(dist[j]), 1), "match_fit": round(float(fit[j]), 2)})
    return pd.DataFrame(rows)


def main() -> None:
    b = load_bundle()
    eff, scen = marketing_effects(b)
    sup = creator_supply(scen)
    creators = creator_pool()
    ew = early_warning(b)
    m = match(ew, creators)
    pr = priority_districts()
    eff.to_csv(PROC / "program_marketing_effects.csv", index=False)
    scen.to_csv(PROC / "program_scenarios.csv", index=False)
    creators.to_csv(PROC / "program_creators.csv", index=False)
    ew.to_csv(PROC / "program_early_warning.csv", index=False)
    m.to_csv(PROC / "program_matches.csv", index=False)
    pr.to_csv(PROC / "program_priority_districts.csv", index=False)
    program_cost(scen).to_csv(PROC / "program_cost.csv", index=False)
    (REPORTS / "program_supply.json").write_text(json.dumps(sup, indent=2))
    pd.set_option("display.width", 200)
    print("Estimated effect of the marketing package on weak-online businesses, by sector:")
    print(eff[["sector", "relevance", "weak_online_share", "weak_online_businesses", "p_close_12m_now",
               "risk_reduction_pp_low", "risk_reduction_pp", "risk_reduction_pp_high", "relative_reduction_pct",
               "model_upper_bound_relative_pct", "score_uplift"]].round(3).to_string(index=False))
    print("\nProgram scenarios (exits avoided, cumulative active):")
    print(scen.round(0).to_string(index=False))
    print("\nCreator supply:", json.dumps({k: round(v, 1) if isinstance(v, float) else v for k, v in sup.items()}))
    print(f"\nEarly warning: {len(ew)} businesses; matched {len(m)}; avg risk drop {m['risk_drop_pp'].mean():.1f} pp")
    print("\nPriority districts:", pr["district"].tolist())
    print("\nCost per outcome (default voucher AED 1,250/month x 6 months + 15% admin):")
    print(program_cost(scen).round(0).to_string(index=False))
    print("\nExample advisor (F&B, Nuaimeya 1):")
    print(district_opportunity({"sector": "F&B", "area": "Nuaimeya 1"}, b).head(8)
          [["district", "parent", "score", "growth_pct", "saturation", "opportunity", "outlook"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()

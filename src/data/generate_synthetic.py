"""Generate a business-level dataset CALIBRATED to real Ajman open data.

No public dataset follows individual Ajman businesses from licensing to closure, so the
business-level records are simulated. Everything that real data can pin down is taken from it:

  REAL (data/real/*.csv, see src/data/real.py)
  - where businesses are and what they do: DED license counts per district x sector (79 districts, 11 sectors)
  - how many open each year: DED new licenses 2016-2025 (organizers' file); 2010-2015 estimated from DED start years
  - how many leave each year: exits = active(t-1) + new(t) - active(t), 2018-2025 (8.4%-19.0% a year)
  - how risk differs by license type and age: Ajman Chamber 2023 cancellations (trade ~2x professional,
    industrial lowest; young firms higher than 7+ year-old firms)
  - market saturation: computed from the real license counts (src/features/saturation.py)

  ASSUMED (clearly synthetic)
  - firm-level attributes (capital, employees, premises), digital presence, Google reviews, and how
    strongly each affects survival. These drive the demo of the scoring method only.

Method: each business draws E ~ Exp(1) and closes in the first month its cumulative hazard exceeds E.
Year multipliers are then adjusted until the simulated exit rate of the WHOLE active stock (including a
pre-2010 legacy stock, kept only for calibration) matches the real exit rate for 2018-2025.

Run:  python -m src.data.generate_synthetic [--seed 42]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.real import SECTORS
from src.features.saturation import load_areas, saturation_table

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "data" / "raw" / "businesses_synthetic.csv"
CALIB_PATH = ROOT / "data" / "processed" / "calibration_vs_real.csv"
STOCK_PATH = ROOT / "data" / "processed" / "stock_profile_2025.csv"
REAL = ROOT / "data" / "real"

START = pd.Timestamp("2010-01-01")
CUTOFF = pd.Timestamp("2025-12-31")
N_MONTHS = (CUTOFF.year - START.year) * 12 + CUTOFF.month - START.month + 1  # 192

# New licenses 2010-2015: DED "Licenses in Ajman (activities)" start-year counts (still-listed licenses)
# x 1.5 to allow for licenses that have since left (2016 ratio: 4,232 issued vs ~2,800 still listed).
NEW_2010_2015 = {2010: 1205, 2011: 1209, 2012: 1014, 2013: 1090, 2014: 1570, 2015: 1927}
UNLISTED_FACTOR = 1.5

LICENSE_TYPE = {"F&B": "Trade", "retail": "Trade", "trading": "Trade", "real estate": "Trade", "tourism": "Trade",
                "construction": "Professional", "personal services": "Professional", "auto services": "Professional",
                "professional services": "Professional", "tech": "Professional", "manufacturing": "Industrial"}
# Relative hazard by license type: Ajman Chamber 2023 cancellation rates, ages 1-6 (trade 8.0%, professional 3.5%,
# industrial 1.6%) -> 1 : 0.45 : 0.2.  Within-type sector multipliers are ASSUMPTIONS.
TYPE_HAZARD = {"Trade": 1.0, "Professional": 0.45, "Industrial": 0.20}
SECTOR_HAZARD = {"F&B": 1.25, "retail": 1.0, "trading": 0.9, "real estate": 1.0, "tourism": 1.15,
                 "construction": 1.1, "personal services": 1.0, "auto services": 0.9,
                 "professional services": 1.0, "tech": 1.15, "manufacturing": 1.0}
COVID_HIT = {"F&B", "retail", "tourism", "personal services"}


def age_multiplier(age_months: np.ndarray) -> np.ndarray:
    """Chamber 2023 pattern (trade): age 1 = 0.83, 2-3 = 1.0, 4-6 = 0.72, 7+ = 0.38 of the 2-3 year rate."""
    return np.select([age_months < 12, age_months < 36, age_months < 72], [0.85, 1.0, 0.75], 0.42)


EMPLOYEE_MEAN = {"F&B": 6, "retail": 3, "trading": 4, "construction": 12, "personal services": 3,
                 "auto services": 5, "professional services": 3, "tech": 4, "real estate": 3, "tourism": 4,
                 "manufacturing": 18}
PREMISES = {
    "Industrial": (["warehouse", "factory"], [0.5, 0.5]),
    "Trade": (["shop", "office", "warehouse", "flexi-desk"], [0.6, 0.2, 0.1, 0.1]),
    "Professional": (["shop", "office", "flexi-desk", "home-based"], [0.45, 0.3, 0.15, 0.1]),
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def real_annual() -> pd.DataFrame:
    return pd.read_csv(REAL / "annual.csv")


def cohort_sizes() -> dict[int, int]:
    a = real_annual().set_index("year")
    sizes = {y: int(v * UNLISTED_FACTOR) for y, v in NEW_2010_2015.items()}
    sizes.update({int(y): int(a.loc[y, "new"]) for y in range(2016, 2026)})
    return sizes


def _issue_dates(rng, sizes: dict[int, int]) -> pd.DatetimeIndex:
    parts = []
    for y, k in sizes.items():
        days = rng.uniform(0, 365, k)
        parts.append(pd.Timestamp(f"{y}-01-01") + pd.to_timedelta(days, unit="D"))
    return pd.DatetimeIndex(np.concatenate([p.values for p in parts])).normalize()


def _simulate(base_h, lin, issue_m, E, year_mult, covid_hit, legacy_age0=None):
    """Monthly cumulative-hazard simulation. Returns event month index (calendar, -1 = none)."""
    n = len(base_h)
    H = np.zeros(n)
    event_cal = np.full(n, -1)
    alive = np.ones(n, dtype=bool)
    rel = base_h * np.exp(lin)
    for cal in range(N_MONTHS):
        open_ = alive & (issue_m <= cal)
        if not open_.any():
            continue
        age = cal - issue_m + (0 if legacy_age0 is None else legacy_age0)
        year = START.year + cal // 12
        mult = year_mult[year] * np.where(covid_hit & (year == 2020) & (cal % 12 >= 2), 1.5, 1.0)
        H = H + np.where(open_, rel * age_multiplier(age) * mult, 0.0)
        dies = open_ & (H >= E)
        event_cal[dies] = cal
        alive &= ~dies
    return event_cal


def _yearly_stock(issue_m, event_cal, weight=1.0):
    """Active at end of each year and exits during each year (weighted counts)."""
    out = {}
    for y in range(START.year, CUTOFF.year + 1):
        end = (y - START.year) * 12 + 11
        beg = end - 12
        active_end = ((issue_m <= end) & ((event_cal < 0) | (event_cal > end))).sum() * weight
        exits = ((event_cal > beg) & (event_cal <= end)).sum() * weight
        out[y] = (active_end, exits)
    return out


def generate(seed: int = 42, calibrate_iters: int = 8, verbose: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    areas = load_areas()
    real_counts = pd.read_csv(REAL / "district_sector.csv")
    real_counts = real_counts[real_counts["licenses"] > 0]
    sat_tab = saturation_table()  # from REAL license counts
    sat_map = sat_tab.set_index(["area", "sector"])["saturation"]

    # --- cohorts and where/what (real distributions) ---
    sizes = cohort_sizes()
    issue_date = _issue_dates(rng, sizes)
    n = len(issue_date)
    p = real_counts["licenses"].values / real_counts["licenses"].sum()
    pick = rng.choice(len(real_counts), size=n, p=p)
    area = real_counts["district"].values[pick]
    sector = real_counts["sector"].values[pick]
    saturation = np.array([sat_map.get((a, s), 0.0) for a, s in zip(area, sector)])
    issue_m = ((issue_date.year - START.year) * 12 + issue_date.month - 1).values

    # --- firm attributes (assumed) ---
    license_type = np.array([LICENSE_TYPE[s] for s in sector])
    free_zone = rng.random(n) < 0.05  # mainland DED licenses; a few dual-licensed
    legal_form = rng.choice(["sole establishment", "LLC", "branch", "civil company"], size=n, p=[0.55, 0.32, 0.06, 0.07])
    size_factor = np.where(legal_form == "LLC", 1.6, 1.0) * np.where(legal_form == "branch", 2.0, 1.0)
    capital_aed = np.round(rng.lognormal(np.log(120_000), 1.0, n) * size_factor, -3).clip(10_000, 50_000_000)
    emp_mean = np.array([EMPLOYEE_MEAN[s] for s in sector]) * (capital_aed / 120_000) ** 0.3
    employees = np.maximum(1, rng.poisson(emp_mean))
    premises_type = np.empty(n, dtype=object)
    for lt, (opts, pr) in PREMISES.items():
        idx = np.where(license_type == lt)[0]
        premises_type[idx] = rng.choice(opts, size=idx.size, p=pr)

    # --- latent drivers & digital presence in first 6 months (assumed relationships) ---
    m = rng.normal(0, 1, n) + 0.15 * _z(np.log(capital_aed))
    m += np.where(np.isin(sector, ["F&B", "retail", "tourism", "tech", "personal services"]), 0.3, 0.0)
    m -= np.where(np.isin(sector, ["manufacturing", "construction", "trading"]), 0.35, 0.0)
    m += 0.06 * (issue_date.year.values - 2018)  # adoption grows over time
    q = np.where(rng.random(n) < 0.10, rng.normal(-2.2, 1.0, n), rng.normal(0.15, 0.9, n))
    posts_per_month = rng.poisson(np.exp(1.2 + 0.7 * m)).clip(0, 90)
    has_instagram = (posts_per_month > 0) | (rng.random(n) < 0.1)
    real_followers = rng.lognormal(np.log(400) + 0.9 * m + 0.2 * q, 0.8)
    real_engagement = rng.beta(2 + np.maximum(0, 1 + 0.6 * m + 0.6 * q), 60)
    engaged_audience = np.where(has_instagram, real_followers * real_engagement, 0.0)
    inflated = has_instagram & (rng.random(n) < 0.08)
    followers = real_followers * np.where(inflated, rng.uniform(5, 20, n), 1.0)
    instagram_followers = np.where(has_instagram, np.round(followers), 0).astype(int)
    engagement_rate = np.where(has_instagram, engaged_audience / np.maximum(followers, 1), 0.0).round(4)
    days_since_last_post = np.where(posts_per_month > 0, rng.exponential(30 / (1 + posts_per_month / 4)), 180
                                    ).round().astype(int).clip(0, 180)
    has_website = rng.random(n) < _sigmoid(-0.5 + 0.9 * m)
    has_tiktok = rng.random(n) < _sigmoid(-1.8 + 0.8 * m + 0.25 * (issue_date.year.values - 2020))
    has_whatsapp_business = rng.random(n) < _sigmoid(0.2 + 0.6 * m)
    google_review_count = rng.poisson(np.exp(1.6 + 0.7 * m + 0.15 * q + 0.35 * np.maximum(-q, 0))).clip(0, 5000)
    rating_noise = rng.normal(0, 1, n) / np.sqrt(np.maximum(google_review_count, 1))
    google_rating = np.where(google_review_count > 0, np.clip(4.1 + 0.7 * q + rating_noise, 1.0, 5.0), np.nan).round(1)
    responds_to_reviews = (google_review_count > 0) & (rng.random(n) < _sigmoid(-0.5 + 0.8 * m + 0.3 * q))

    lin = (
        -0.25 * _z(np.log(capital_aed)) - 0.20 * _z(np.log1p(employees))
        + 1.20 * saturation
        - 0.45 * q - 0.12 * m
        - 0.20 * _z(np.log1p(posts_per_month)) - 0.10 * np.exp(-days_since_last_post / 30) - 0.08 * has_website
        - 0.15 * _z(np.log1p(engaged_audience))
        - 0.10 * free_zone + np.where(premises_type == "home-based", 0.2, 0.0)
        + rng.normal(0, 0.25, n)
    )
    lin = lin - np.log(np.mean(np.exp(lin)))  # mean relative risk = 1, so the level is set by calibration
    base_h = np.array([TYPE_HAZARD[LICENSE_TYPE[s]] * SECTOR_HAZARD[s] for s in sector])
    covid = np.isin(sector, list(COVID_HIT))
    E = rng.exponential(1.0, n)

    # --- legacy stock (licensed before 2010) - only to make the active stock realistic for calibration ---
    nL = 20_000
    L_sector = real_counts["sector"].values[rng.choice(len(real_counts), size=nL, p=p)]
    L_base = np.array([TYPE_HAZARD[LICENSE_TYPE[s]] * SECTOR_HAZARD[s] for s in L_sector])
    L_age0 = rng.integers(12, 240, nL)  # 1-20 years old in Jan 2010
    L_lin = rng.normal(-0.3, 0.6, nL)   # survivors are selected for lower risk
    L_E = rng.exponential(1.0, nL)
    L_issue = np.zeros(nL, dtype=int)
    L_covid = np.isin(L_sector, list(COVID_HIT))

    real = real_annual().set_index("year")
    target = real["exit_rate"].dropna()  # 2018-2025
    year_mult = {y: 0.012 for y in range(START.year, CUTOFF.year + 1)}  # monthly base level
    w_legacy = 1.0
    for it in range(calibrate_iters):
        ev = _simulate(base_h, lin, issue_m, E, year_mult, covid)
        evL = _simulate(L_base, L_lin, L_issue, L_E, year_mult, L_covid, legacy_age0=L_age0)
        s_new, s_leg = _yearly_stock(issue_m, ev), _yearly_stock(L_issue, evL)
        # legacy weight so total active at end-2017 equals the real 29,909
        need = real.loc[2017, "active"] - s_new[2017][0]
        w_legacy = max(need, 0) / max(s_leg[2017][0], 1)
        rates = {}
        for y in target.index:
            act_prev = s_new[y - 1][0] + w_legacy * s_leg[y - 1][0]
            ex = s_new[y][1] + w_legacy * s_leg[y][1]
            rates[y] = ex / act_prev
        for y in target.index:
            year_mult[y] *= float(np.clip(target[y] / rates[y], 0.5, 2.0)) ** 0.9
        for y in range(START.year, 2018):
            year_mult[y] = year_mult[2018]
        if verbose:
            err = np.mean([abs(rates[y] - target[y]) for y in target.index])
            print(f"  calibration iter {it + 1}: mean |sim - real| exit rate = {err:.4f}")

    # final run with calibrated multipliers
    ev = _simulate(base_h, lin, issue_m, E, year_mult, covid)
    evL = _simulate(L_base, L_lin, L_issue, L_E, year_mult, L_covid, legacy_age0=L_age0)
    s_new, s_leg = _yearly_stock(issue_m, ev), _yearly_stock(L_issue, evL)
    calib = []
    for y in range(2017, CUTOFF.year + 1):
        act = s_new[y][0] + w_legacy * s_leg[y][0]
        ex = s_new[y][1] + w_legacy * s_leg[y][1]
        prev = s_new[y - 1][0] + w_legacy * s_leg[y - 1][0]
        calib.append({"year": y, "sim_active": round(act), "real_active": real.loc[y, "active"],
                      "sim_exits": round(ex), "real_exits": real.loc[y, "exits"],
                      "sim_exit_rate": ex / prev, "real_exit_rate": real.loc[y, "exit_rate"],
                      "year_multiplier": year_mult[y]})
    calib = pd.DataFrame(calib)
    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    calib.to_csv(CALIB_PATH, index=False)

    # age/sector profile of the simulated active stock at end-2025 (used by the forecast module)
    end = N_MONTHS - 1
    act_new = (ev < 0)
    act_leg = (evL < 0)
    prof = pd.concat([
        pd.DataFrame({"sector": sector[act_new], "age_months": end - issue_m[act_new], "weight": 1.0,
                      "rel_risk": (base_h * np.exp(lin))[act_new]}),
        pd.DataFrame({"sector": L_sector[act_leg], "age_months": end + L_age0[act_leg], "weight": w_legacy,
                      "rel_risk": (L_base * np.exp(L_lin))[act_leg]}),
    ])
    prof.to_csv(STOCK_PATH, index=False)
    pd.Series(year_mult).rename("monthly_level").to_csv(ROOT / "data" / "processed" / "year_multipliers.csv")

    event = (ev >= 0).astype(int)
    months_open = (N_MONTHS - 1 - issue_m) + 1
    duration_months = np.where(event == 1, ev - issue_m + 1, months_open)
    cancel_date = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
    idx = np.where(event == 1)[0]
    cancel_date.iloc[idx] = [max(START + pd.DateOffset(months=int(c)) + pd.Timedelta(days=14), d + pd.Timedelta(days=7))
                             for c, d in zip(ev[idx], issue_date[idx])]
    renewals = (duration_months // 12).astype(int)
    last_renewal_date = pd.DatetimeIndex([d + pd.DateOffset(years=int(r)) for d, r in zip(issue_date, renewals)])

    return pd.DataFrame({
        "license_id": [f"AJM-{i:06d}" for i in range(1, n + 1)],
        "issue_date": issue_date.date, "cancel_date": cancel_date.dt.date, "last_renewal_date": last_renewal_date.date,
        "sector": sector, "area": area, "license_type": license_type, "legal_form": legal_form,
        "capital_aed": capital_aed.astype(int), "employees": employees, "free_zone": free_zone,
        "premises_type": premises_type,
        "has_website": has_website, "google_rating": google_rating, "google_review_count": google_review_count,
        "instagram_followers": instagram_followers, "posts_per_month": posts_per_month,
        "engagement_rate": engagement_rate, "days_since_last_post": days_since_last_post,
        "has_tiktok": has_tiktok, "has_whatsapp_business": has_whatsapp_business,
        "responds_to_reviews": responds_to_reviews,
        "duration_months": duration_months, "event": event, "is_synthetic": True,
    })


def sanity_report(df: pd.DataFrame) -> None:
    cutoff36 = (CUTOFF - pd.DateOffset(months=36)).date()
    c = df[pd.to_datetime(df["issue_date"]).dt.date <= cutoff36].copy()
    c["alive_36m"] = ~((c["event"] == 1) & (c["duration_months"] <= 36))
    print(f"Rows: {len(df):,} | closed: {df['event'].sum():,} ({df['event'].mean():.1%})")
    print("\nCalibration vs REAL (DED annual data):")
    print(pd.read_csv(CALIB_PATH).round(3).to_string(index=False))
    print("\n3-year survival by sector:")
    print(c.groupby("sector")["alive_36m"].mean().sort_values().map("{:.1%}".format).to_string())
    print("\n3-year survival by license type:")
    print(c.groupby("license_type")["alive_36m"].mean().map("{:.1%}".format).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    df = generate(args.seed)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH.relative_to(ROOT)}\n")
    sanity_report(df)


if __name__ == "__main__":
    main()

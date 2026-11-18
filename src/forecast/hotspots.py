"""Sector and district growth forecasts ("hotspots"), 2026-2028, plus free zone forecasts by license type.

Entries:  New_{k,t} = share_{k,t} x New_t
          share_{k,t} follows each sector's / district's recent trend in its share of new licenses (damped),
          New_t is the emirate-wide forecast from src/forecast/forecast.py.
Exits:    Exits_{k,t} = sum over active businesses in k of the survival model's hazard (sectors), or
          the model's district closure rate x district stock, rescaled to the emirate-wide exit forecast (districts).
Growth:   Active_{k,t+1} = Active_{k,t} + New_{k,t+1} - Exits_{k,t+1};  hotspot = growth vs Ajman average.

Real inputs: DED license starts by sector (2016-2023 complete) and by district (2019-2023, 2025 complete;
2024 is incomplete in the open data and skipped), DED license stock per district, AFZ monthly registrations.

Run:  python -m src.forecast.hotspots     (after src.forecast.forecast)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.data.load import ROOT
from src.data.real import SECTORS, key
from src.forecast.forecast import HORIZON_YEARS, project_stock

REAL = ROOT / "data" / "real"
RAW = REAL / "raw"
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
DAMP = 0.5  # keep half of the recent share trend per year (avoids runaway extrapolation)


def _starts() -> tuple[pd.DataFrame, pd.DataFrame]:
    m = dict(line.split("=", 1) for line in (RAW / "starts_by_year.txt").read_text().strip().splitlines())
    sy = pd.DataFrame([r.split("|") for r in m["sector_year"].split(";")], columns=["sector", "year", "n"])
    dy = pd.DataFrame([r.split("|") for r in m["district_year"].split(";")], columns=["raw", "year", "n"])
    for d in (sy, dy):
        d["year"] = d["year"].astype(int)
        d["n"] = d["n"].astype(int)
    return sy, dy


def _project_shares(shares: pd.DataFrame, years_fit: list[int], base_year: int) -> pd.DataFrame:
    """shares: index=group, columns=years. Damped log-linear trend from years_fit, starting at base_year's share."""
    s = shares[years_fit].clip(lower=1e-4)
    x = np.array(years_fit, dtype=float)
    slope = np.array([np.polyfit(x, np.log(row.values), 1)[0] for _, row in s.iterrows()])
    slope = np.clip(slope, -0.35, 0.35)
    base = shares[base_year].clip(lower=1e-4).values
    out = {}
    for i, y in enumerate(HORIZON_YEARS, start=1):
        cum = sum(DAMP ** j for j in range(1, i + 1 + (y - base_year - i)))  # years since base, damped
        v = base * np.exp(slope * cum)
        out[y] = v / v.sum()
    res = pd.DataFrame(out, index=shares.index)
    res["trend_pct_per_year"] = 100 * (np.exp(slope) - 1)
    return res


def sector_outlook(fc: pd.DataFrame) -> pd.DataFrame:
    sy, _ = _starts()
    sy = sy[sy["sector"].isin(SECTORS)]
    wide = sy.pivot_table(index="sector", columns="year", values="n", aggfunc="sum").fillna(0)
    shares = wide / wide.sum()
    proj = _project_shares(shares, [2019, 2020, 2021, 2022, 2023], 2023)

    stock = pd.read_csv(PROC / "stock_profile_2025.csv")
    levels = pd.read_csv(PROC / "year_multipliers.csv", index_col=0)["monthly_level"]
    a = pd.read_csv(REAL / "annual.csv")
    active0 = float(a.loc[a["year"] == 2025, "active"].iloc[0])
    w_tot = stock["weight"].sum()
    rows = []
    for s in SECTORS:
        st = stock[stock["sector"] == s]
        act_s = active0 * st["weight"].sum() / w_tot
        new = {y: float(proj.loc[s, y] * fc.loc[fc["year"] == y, "new"].iloc[0]) for y in HORIZON_YEARS}
        p = project_stock(act_s, new, float(levels.loc[2025]), st, pd.Series({s: 1.0}))
        rows.append({"sector": s, "active_2025": act_s, "share_new_2023": shares.loc[s, 2023],
                     "share_new_2028": proj.loc[s, 2028], "share_trend_pct_yr": proj.loc[s, "trend_pct_per_year"],
                     **{f"new_{y}": new[y] for y in HORIZON_YEARS},
                     **{f"exits_{y}": float(p.loc[p["year"] == y, "exits"].iloc[0]) for y in HORIZON_YEARS},
                     "active_2028": float(p["active"].iloc[-1])})
    out = pd.DataFrame(rows)
    # reconcile exits/active with the emirate-wide forecast
    scale = fc.set_index("year")["active"].iloc[-1] / out["active_2028"].sum()
    out["active_2028"] *= scale
    out["growth_2025_2028_pct"] = 100 * (out["active_2028"] / out["active_2025"] - 1)
    avg = 100 * (out["active_2028"].sum() / out["active_2025"].sum() - 1)
    out["vs_ajman_pp"] = out["growth_2025_2028_pct"] - avg
    out["outlook"] = np.select([out["vs_ajman_pp"] > 5, out["vs_ajman_pp"] < -5], ["Rising", "Cooling"], "In line")
    sat = pd.read_csv(PROC / "saturation_current.csv")
    out["avg_saturation"] = out["sector"].map(sat.groupby("sector").apply(
        lambda g: np.average(g["saturation"], weights=g["n"]) if g["n"].sum() else 0, include_groups=False))
    return out.sort_values("growth_2025_2028_pct", ascending=False)


def district_outlook(fc: pd.DataFrame) -> pd.DataFrame:
    _, dy = _starts()
    dist = pd.read_csv(REAL / "districts.csv")
    dy["key"] = dy["raw"].map(key)
    dy = dy[dy["key"].isin(dist["key"])]
    wide = dy.pivot_table(index="key", columns="year", values="n", aggfunc="sum").reindex(dist["key"]).fillna(0)
    wide = wide[[c for c in [2019, 2020, 2021, 2022, 2023, 2025, 2026] if c in wide.columns]]
    shares = (wide + 0.5) / (wide + 0.5).sum()
    # momentum: 2023 -> 2025 (two years) and 2025 -> H1 2026; fit on 2021, 2022, 2023, 2025
    s_fit = shares[[2021, 2022, 2023, 2025]]
    proj = _project_shares(s_fit, [2021, 2022, 2023, 2025], 2025)
    area = pd.read_csv(PROC / "area_stats.csv").set_index("area")
    d = dist.set_index("key")
    rows = []
    exits_tot = fc.set_index("year")["exits"]
    for k, r in d.iterrows():
        act = float(r["licenses_listed"])
        rate = float(area.loc[r["district"], "forecast_closure_rate"]) if r["district"] in area.index else 0.0
        recent = float(wide.loc[k, [c for c in (2025, 2026) if c in wide.columns]].sum())
        rows.append({"district": r["district"], "key": k, "parent": r["parent"], "city": r["city"], "active_2025": act,
                     "starts_2023": float(wide.loc[k, 2023]), "starts_2025_26": recent,
                     "rate": 0.0 if np.isnan(rate) else rate, "share_new_2025": shares.loc[k, 2025],
                     "share_new_h1_2026": shares.loc[k, 2026] if 2026 in shares.columns else np.nan,
                     "share_trend_pct_yr": proj.loc[k, "trend_pct_per_year"],
                     **{f"share_{y}": proj.loc[k, y] for y in HORIZON_YEARS}})
    out = pd.DataFrame(rows)
    out = out[out["active_2025"] > 0].copy()
    out[[f"share_{y}" for y in HORIZON_YEARS]] /= out[[f"share_{y}" for y in HORIZON_YEARS]].sum()
    active = out["active_2025"].values.copy()
    for y in HORIZON_YEARS:
        new = out[f"share_{y}"].values * float(fc.loc[fc["year"] == y, "new"].iloc[0])
        raw_ex = out["rate"].values * active
        ex = raw_ex * float(exits_tot.loc[y]) / max(raw_ex.sum(), 1e-9)
        out[f"new_{y}"] = new
        out[f"exits_{y}"] = ex
        active = active + new - ex
        out[f"active_{y}"] = active
    out["growth_2025_2028_pct"] = 100 * (out["active_2028"] / out["active_2025"] - 1)
    avg = 100 * (out["active_2028"].sum() / out["active_2025"].sum() - 1)
    out["vs_ajman_pp"] = out["growth_2025_2028_pct"] - avg
    small = out["active_2025"] < 50
    # a district with many starts in 2023 but none in 2025-26 most likely changed name in the register
    renamed = (out["starts_2023"] >= 30) & (out["starts_2025_26"] == 0)
    out["outlook"] = np.select([small, renamed, out["vs_ajman_pp"] > 8, out["vs_ajman_pp"] < -8],
                               ["Too small to call", "Check data (no 2025 starts)", "Hotspot", "Cooling"], "In line")
    return out.sort_values("growth_2025_2028_pct", ascending=False)


def afz_by_type(h: int = 12) -> pd.DataFrame:
    m = pd.read_csv(REAL / "afz_monthly.csv", parse_dates=["month"])
    m = m[~m["partial_month"]]
    tot_fc = pd.read_csv(PROC / "forecast_monthly_afz.csv")["new_companies"].values
    rows = []
    for c in [c for c in m.columns if c.startswith("License:")]:
        y = m[c].astype(float).values
        if y.sum() < 50:
            f = np.full(h, y[-6:].mean())
        else:
            fit = ExponentialSmoothing(np.log1p(y), trend="add", damped_trend=True).fit()
            f = np.expm1(fit.forecast(h)).clip(min=0)
            f = 0.5 * f + 0.5 * y[-3:].mean()
        rows.append({"license_type": c.replace("License: ", ""), "last_12m": y[-12:].sum(), "raw_next_12m": f.sum(),
                     "share_last_12m": y[-12:].sum() / m["new_companies"].iloc[-12:].sum()})
    out = pd.DataFrame(rows)
    out["next_12m"] = out["raw_next_12m"] * tot_fc.sum() / out["raw_next_12m"].sum()  # reconcile to total
    out["growth_pct"] = 100 * (out["next_12m"] / out["last_12m"].clip(lower=1) - 1)
    return out.sort_values("next_12m", ascending=False)


def main() -> None:
    fc = pd.read_csv(PROC / "forecast_annual.csv")
    so, do, af = sector_outlook(fc), district_outlook(fc), afz_by_type()
    so.to_csv(PROC / "forecast_sectors.csv", index=False)
    do.to_csv(PROC / "forecast_districts.csv", index=False)
    af.to_csv(PROC / "forecast_afz_by_type.csv", index=False)
    pd.set_option("display.width", 200)
    print("Sector outlook 2025 -> 2028:")
    print(so[["sector", "active_2025", "new_2026", "exits_2026", "active_2028", "growth_2025_2028_pct",
              "share_trend_pct_yr", "outlook"]].round(1).to_string(index=False))
    print("\nTop district hotspots:")
    cols = ["district", "parent", "active_2025", "new_2026", "growth_2025_2028_pct", "outlook"]
    print(do[do["outlook"] != "Too small to call"][cols].head(10).round(1).to_string(index=False))
    print("\nCooling districts:")
    print(do[do["outlook"] == "Cooling"][cols].tail(8).round(1).to_string(index=False))
    print("\nFree zone by license type (next 12 months):")
    print(af.round(1).to_string(index=False))
    summary = {"rising_sectors": so.loc[so["outlook"] == "Rising", "sector"].tolist(),
               "cooling_sectors": so.loc[so["outlook"] == "Cooling", "sector"].tolist(),
               "hotspot_districts": do.loc[do["outlook"] == "Hotspot", "district"].tolist(),
               "cooling_districts": do.loc[do["outlook"] == "Cooling", "district"].tolist()}
    (REPORTS / "forecast_hotspots.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

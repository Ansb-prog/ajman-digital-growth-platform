"""Forecasts of Ajman business activity.

1) Annual license stock (DED, 2016-2025 real) -> 2026-2028, with the stock-flow equation

       Active(t+1) = Active(t) + New(t+1) - Exits(t+1)
       Exits(t+1)  = sum over active businesses i of  h_i(age_i, sector_i) * L(t+1)

   New licenses: average of a log-linear trend (2020-2025) and a damped Holt model (2016-2025).
   Exits: the survival model's hazards applied month by month to the real-size active stock (age and sector
   profile from the calibrated simulation), at the current exit-risk level L (mean of 2023-2025) with
   low/high scenarios from the 2022-2025 range. This ties the firm-level score to the emirate-level forecast.

2) Monthly Ajman Free Zone new companies (organizers' file, Jan 2024 - Oct 2026 complete months) -> next 12
   months, with exponential smoothing on log counts (trend damped; seasonal only if it beats the backtest).

Both are backtested against simple baselines. Run:  python -m src.forecast.forecast
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.data.generate_synthetic import COVID_HIT, LICENSE_TYPE, SECTOR_HAZARD, TYPE_HAZARD, age_multiplier
from src.data.load import ROOT

warnings.filterwarnings("ignore")
REAL = ROOT / "data" / "real"
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
HORIZON_YEARS = [2026, 2027, 2028]


# ------------------------------------------------------------------ annual
def forecast_new(a: pd.DataFrame, years: list[int], n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """New licenses: ensemble of log-linear trend (2020+) and damped Holt (all years), bootstrap intervals."""
    rng = np.random.default_rng(seed)
    hist = a.dropna(subset=["new"]).set_index("year")["new"].astype(float)
    recent = hist[hist.index >= 2020]
    b1, b0 = np.polyfit(recent.index, np.log(recent.values), 1)
    trend = {y: np.exp(b0 + b1 * y) for y in years}
    resid_t = np.log(recent.values) - (b0 + b1 * recent.index.values)
    holt = ExponentialSmoothing(np.log(hist.values), trend="add", damped_trend=True).fit()
    h = np.exp(holt.forecast(len(years)))
    point = {y: 0.5 * trend[y] + 0.5 * h[i] for i, y in enumerate(years)}
    sd = max(np.std(resid_t, ddof=1), np.std(holt.resid, ddof=1))
    sims = np.array([[point[y] * np.exp(rng.normal(0, sd * np.sqrt(i + 1))) for i, y in enumerate(years)]
                     for _ in range(n_boot)])
    return pd.DataFrame({"year": years, "new": [point[y] for y in years],
                         "new_lo": np.percentile(sims, 10, axis=0), "new_hi": np.percentile(sims, 90, axis=0),
                         "growth_trend_pct": 100 * (np.exp(b1) - 1)})


def project_stock(active0: float, new_by_year: dict[int, float], level: float, stock: pd.DataFrame,
                  sector_mix: pd.Series) -> pd.DataFrame:
    """Monthly stock-flow projection with model hazards. `stock` = age/sector/rel_risk/weight of active firms."""
    w = stock["weight"].values * active0 / stock["weight"].sum()  # scale simulated stock to real size
    age = stock["age_months"].values.astype(float)
    rel = stock["rel_risk"].values
    sec = stock["sector"].values
    covid = np.isin(sec, list(COVID_HIT))  # no covid shock in the future; kept for shape
    _ = covid
    new_base = float(sum(TYPE_HAZARD[LICENSE_TYPE[s]] * SECTOR_HAZARD[s] * p for s, p in sector_mix.items()))
    cohorts_w, cohorts_age = [], []
    rows = []
    active = active0
    for y, new_y in new_by_year.items():
        exits_y = 0.0
        for _m in range(12):
            h = 1 - np.exp(-rel * age_multiplier(age) * level)
            ex = (w * h).sum()
            w = w * (1 - h)
            age = age + 1
            # new cohorts (one per month), average relative risk 1 within sector mix
            cohorts_w.append(new_y / 12)
            cohorts_age.append(0.0)
            cw, ca = np.array(cohorts_w), np.array(cohorts_age)
            hn = 1 - np.exp(-new_base * age_multiplier(ca) * level)
            ex += (cw * hn).sum()
            cohorts_w = list(cw * (1 - hn))
            cohorts_age = list(ca + 1)
            exits_y += ex
        active = w.sum() + sum(cohorts_w)
        rows.append({"year": y, "new": new_y, "exits": exits_y, "active": active})
    return pd.DataFrame(rows)


def annual_forecast() -> tuple[pd.DataFrame, dict]:
    a = pd.read_csv(REAL / "annual.csv")
    stock = pd.read_csv(PROC / "stock_profile_2025.csv")
    levels = pd.read_csv(PROC / "year_multipliers.csv", index_col=0)["monthly_level"]
    mix = pd.read_csv(REAL / "district_sector.csv").groupby("sector")["licenses"].sum()
    mix = mix / mix.sum()
    # exit-risk level: latest year (2025) as central case; low = the 2023->2025 decline continues; high = 2022-2024 mean
    level_mid = float(levels.loc[2025])
    level_lo = float(levels.loc[2025] * (levels.loc[2025] / levels.loc[2023]) ** 0.5)
    level_hi = float(levels.loc[[2022, 2023, 2024]].mean())

    nf = forecast_new(a, HORIZON_YEARS)
    active0 = float(a.loc[a["year"] == 2025, "active"].iloc[0])
    out = []
    for name, lvl, col in [("mid", level_mid, "new"), ("low_exits_high_entry", level_lo, "new_hi"),
                           ("high_exits_low_entry", level_hi, "new_lo")]:
        p = project_stock(active0, dict(zip(nf["year"], nf[col])), lvl, stock, mix)
        p["scenario"] = name
        out.append(p)
    proj = pd.concat(out)
    mid = proj[proj["scenario"] == "mid"].set_index("year")
    rng_ = proj.groupby("year")[["exits", "active"]].agg(["min", "max"])
    fc = pd.DataFrame({
        "year": HORIZON_YEARS,
        "new": nf["new"].values, "new_lo": nf["new_lo"].values, "new_hi": nf["new_hi"].values,
        "exits": mid["exits"].values, "exits_lo": rng_[("exits", "min")].values, "exits_hi": rng_[("exits", "max")].values,
        "active": mid["active"].values, "active_lo": rng_[("active", "min")].values, "active_hi": rng_[("active", "max")].values,
    })
    fc["exit_rate"] = fc["exits"] / np.r_[active0, fc["active"].values[:-1]]

    # ---- backtest: pretend we are at end-2023, forecast 2024-2025 ----
    a23 = a[a["year"] <= 2023]
    nf_bt = forecast_new(a23, [2024, 2025])
    lvl_bt = float(levels.loc[2023])  # latest level known at the backtest origin
    act23 = float(a.loc[a["year"] == 2023, "active"].iloc[0])
    p_bt = project_stock(act23, dict(zip(nf_bt["year"], nf_bt["new"])), lvl_bt, stock, mix)
    actual = a.set_index("year").loc[[2024, 2025]]
    naive_growth = act23 / float(a.loc[a["year"] == 2022, "active"].iloc[0])
    naive_active = [act23 * naive_growth, act23 * naive_growth ** 2]
    mape = lambda f, t: float(np.mean(np.abs(np.array(f) - np.array(t)) / np.array(t)))  # noqa: E731
    bt = {
        "annual_backtest_origin": 2023,
        "active_forecast": [round(v) for v in p_bt["active"]], "active_actual": actual["active"].astype(int).tolist(),
        "active_mape_model": mape(p_bt["active"], actual["active"]),
        "active_mape_naive_growth": mape(naive_active, actual["active"]),
        "new_forecast": [round(v) for v in nf_bt["new"]], "new_actual": actual["new"].astype(int).tolist(),
        "new_mape_model": mape(nf_bt["new"], actual["new"]),
        "new_mape_naive_last": mape([a23["new"].iloc[-1]] * 2, actual["new"]),
        "exits_forecast": [round(v) for v in p_bt["exits"]], "exits_actual": actual["exits"].astype(int).tolist(),
        "exits_mape_model": mape(p_bt["exits"], actual["exits"]),
        "exits_mape_naive_last": mape([a23["exits"].iloc[-1]] * 2, actual["exits"]),
        "exit_level_mid": level_mid, "new_growth_trend_pct": float(nf["growth_trend_pct"].iloc[0]),
    }
    return fc, bt


# ------------------------------------------------------------------ monthly (Ajman Free Zone)
def _fit_monthly(y: pd.Series, seasonal: bool):
    return ExponentialSmoothing(np.log(y.values), trend="add", damped_trend=True,
                                seasonal="add" if seasonal else None,
                                seasonal_periods=12 if seasonal else None).fit()


def monthly_forecast(h: int = 12) -> tuple[pd.DataFrame, dict]:
    m = pd.read_csv(REAL / "afz_monthly.csv", parse_dates=["month"])
    full = m[~m["partial_month"]].set_index("month")["new_companies"].astype(float)
    # backtest: last 6 complete months
    train, test = full.iloc[:-6], full.iloc[-6:]
    res = {}
    for seasonal in (False, True):
        try:
            f = np.exp(_fit_monthly(train, seasonal).forecast(6))
            res[seasonal] = float(np.mean(np.abs(f - test.values) / test.values))
        except Exception:  # noqa: BLE001 - not enough cycles for seasonal fit
            res[seasonal] = np.inf
    f_ets = np.exp(_fit_monthly(train, False).forecast(6))
    f_ens = 0.5 * f_ets + 0.5 * train.iloc[-3:].mean()
    res["ensemble"] = float(np.mean(np.abs(f_ens - test.values) / test.values))
    naive = float(np.mean(np.abs(train.iloc[-1] - test.values) / test.values))
    naive3 = float(np.mean(np.abs(train.iloc[-3:].mean() - test.values) / test.values))
    use_seasonal = res[True] < res[False]
    fit = _fit_monthly(full, use_seasonal)
    point = np.exp(fit.forecast(h))
    use_ensemble = res["ensemble"] < min(res[False], res[True])
    if use_ensemble:  # blend the trend model with the recent 3-month level (better in backtest)
        point = 0.5 * point + 0.5 * full.iloc[-3:].mean()
    sd = float(np.std(fit.resid, ddof=1))
    rng = np.random.default_rng(1)
    sims = np.exp(np.log(point)[None, :] + rng.normal(0, sd, (2000, h)).cumsum(axis=1) / np.sqrt(np.arange(1, h + 1))
                  * np.sqrt(np.arange(1, h + 1)) ** 0.5)
    idx = pd.date_range(full.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    fc = pd.DataFrame({"month": idx, "new_companies": point, "lo": np.percentile(sims, 10, axis=0),
                       "hi": np.percentile(sims, 90, axis=0)})
    lic_cols = [c for c in m.columns if c.startswith("License:")]
    share = m[~m["partial_month"]].tail(6)[lic_cols].sum() / m[~m["partial_month"]].tail(6)["new_companies"].sum()
    for c in lic_cols:
        fc[c] = fc["new_companies"] * share[c]
    bt = {"monthly_backtest_months": 6, "mape_ets": res[False], "mape_ets_seasonal": res[True],
          "mape_ensemble": res["ensemble"], "ensemble_used": bool(use_ensemble),
          "mape_naive_last": naive, "mape_naive_3m_avg": naive3, "seasonal_used": bool(use_seasonal),
          "next_12m_total": float(point.sum()),
          "last_12m_total": float(full.iloc[-12:].sum())}
    return fc, bt


def main() -> None:
    fc_a, bt_a = annual_forecast()
    fc_m, bt_m = monthly_forecast()
    PROC.mkdir(parents=True, exist_ok=True)
    fc_a.to_csv(PROC / "forecast_annual.csv", index=False)
    fc_m.to_csv(PROC / "forecast_monthly_afz.csv", index=False)
    (REPORTS / "forecast_backtest.json").write_text(json.dumps({**bt_a, **bt_m}, indent=2, default=float))
    pd.set_option("display.width", 160)
    print("Annual forecast (DED licenses):")
    print(fc_a.round(0).to_string(index=False))
    print("\nBacktest (origin 2023):", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in bt_a.items()}))
    print("\nMonthly AFZ forecast:")
    print(fc_m[["month", "new_companies", "lo", "hi"]].round(0).to_string(index=False))
    print("\nBacktest (last 6 months):", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in bt_m.items()}))


if __name__ == "__main__":
    main()

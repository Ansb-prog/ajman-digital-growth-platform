"""Train, evaluate and package the Business Survival Score.

Pipeline
  1. Time-based split: train (licensed <= Jun 2020), validation (Jul 2020-2021), test (2022+)
  2. Fit features (Reputation, learned Digital Presence weights, demand-based saturation) on train
  3. Fit the Cox model with all of them as covariates:
        S(t | x) = S0(t) ^ exp(beta . x + gamma_R R + gamma_D D + gamma_S Saturation)
  4. Evaluate on test: C-index, IPCW Brier at 12/36 months, calibration, PH check, XGBoost baseline,
     and the earlier "multiplicative saturation penalty" variant for comparison
  5. Refit on all data for deployment, then precompute map/forecast tables

Run:  python -m src.models.train
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.load import ROOT, load_businesses
from src.features.build_features import DEPLOY_START, D_COMPONENTS, D_LABELS, FeatureBuilder, digital_components, split
from src.features.saturation import active_at, add_saturation, saturation_table
from src.models.cox import fit_cox, ph_check, survival_at
from src.models.evaluate import brier_ipcw, c_index, calibration_table, censoring_km
from src.models.score import Bundle, save_bundle, score_frame
from src.models.xgb_baseline import fit_xgb, label_36m

REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"
LAMBDAS = np.round(np.arange(0, 1.0001, 0.05), 2)


def penalty_variant(fb, train, val, test) -> dict:
    """Earlier design: Cox WITHOUT saturation, then Score = 100 * S * (1 - lambda * Saturation),
    lambda chosen on validation by C-index. Kept only to show why saturation now lives inside the model."""
    cols = lambda d: fb.transform(d).drop(columns=["saturation"])  # noqa: E731
    cox = fit_cox(cols(train), train["duration_months"], train["event"])
    def c_at(d, lam):
        s = survival_at(cox, cols(d), 36)
        return c_index(d["duration_months"], 100 * s * (1 - lam * fb.saturation_of(d)), d["event"])
    grid = pd.DataFrame({"lambda": LAMBDAS, "c_index": [c_at(val, l) for l in LAMBDAS]})
    lam = float(grid.loc[grid["c_index"].idxmax(), "lambda"])
    return {"lambda": lam, "c_index_test": c_at(test, lam), "grid": grid}


def xgb_matrix(fb: FeatureBuilder, df: pd.DataFrame) -> pd.DataFrame:
    x = fb.transform(df).drop(columns=["digital_presence"])
    return pd.concat([x, digital_components(df).set_index(x.index)], axis=1)


def evaluate_on_test(cox, fb, train, test) -> tuple[dict, pd.DataFrame]:
    from lifelines import KaplanMeierFitter

    te = test
    x = fb.transform(te)
    g = censoring_km(train["duration_months"], train["event"])
    km = KaplanMeierFitter().fit(train["duration_months"], train["event"])
    m: dict = {"n_train": len(train), "n_test": len(te)}
    s36 = survival_at(cox, x, 36)
    m["c_index_score"] = c_index(te["duration_months"], s36, te["event"])
    for t in (12, 36):
        m[f"brier_{t}m"] = brier_ipcw(t, te["duration_months"], te["event"], survival_at(cox, x, t), g)
        # reference: predicting the same overall survival for everyone
        m[f"brier_{t}m_null"] = brier_ipcw(t, te["duration_months"], te["event"],
                                           np.full(len(te), float(km.survival_function_at_times(t).iloc[0])), g)
    cal = calibration_table(s36, te["duration_months"], te["event"], 36)

    # XGBoost vs Cox on test rows with a known 36-month outcome
    xgb = fit_xgb(xgb_matrix(fb, train), label_36m(train))
    y = label_36m(te)
    known = y.notna().values
    m["n_test_known_36m"] = int(known.sum())
    if known.sum() > 50 and y[known].nunique() == 2:
        p_xgb = xgb.predict_proba(xgb_matrix(fb, te)[known])[:, 1]
        m["auc36_xgboost"] = float(roc_auc_score(y[known], p_xgb))
        m["auc36_score"] = float(roc_auc_score(y[known], s36[known]))
    return m, cal


def plot_calibration(cal: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.plot(cal["predicted"], cal["observed"], "o-", lw=1.8)
    ax.set(xlim=(0.3, 1), ylim=(0.3, 1), xlabel="Predicted 3-year survival", ylabel="Observed (Kaplan-Meier)",
           title="Calibration on test cohort (2022+)")
    ax.grid(alpha=0.25)
    fig.text(0.99, 0.01, "Synthetic data", ha="right", fontsize=7, color="grey")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def precompute_tables(b: Bundle, df: pd.DataFrame) -> None:
    """District/sector tables for the dashboard.

    REAL: district polygons, license counts, new licenses, saturation (from real counts), residents (estimated).
    MODEL: scores and 12-month closure probabilities of the simulated active stock, which matches the real
    active stock in size (41k, end-2025) and in district x sector mix; counts are scaled to real listed licenses.
    """
    act = df[df["event"] == 0].copy().reset_index(drop=True)
    sc = score_frame(b, act)
    act = pd.concat([act.drop(columns=["saturation"], errors="ignore"), sc], axis=1)
    x = b.features.transform(act)
    age = act["duration_months"].values
    s_now, s_next = survival_at(b.cox, x, age), survival_at(b.cox, x, age + 12)
    act["p_close_12m"] = 1 - s_next / np.clip(s_now, 1e-9, None)

    districts = pd.read_csv(ROOT / "data" / "real" / "districts.csv").rename(columns={"district": "area"})
    sat = b.saturation_table
    by = act.groupby("area").agg(sim_active=("license_id", "size"), avg_score=("score", "mean"),
                                 avg_digital=("digital_presence", "mean"),
                                 closure_rate_12m=("p_close_12m", "mean"),
                                 top_sectors=("sector", lambda s: ", ".join(s.value_counts().head(3).index)))
    area = districts.merge(by, left_on="area", right_index=True, how="left")
    area["active"] = area["licenses_listed"]  # REAL: DED listed licenses
    area["forecast_closures_12m"] = (area["closure_rate_12m"].fillna(0) * area["active"]).round(1)
    area["forecast_closure_rate"] = area["closure_rate_12m"].fillna(0)
    area["new_licenses_8m"] = area["new_licenses_8m"].fillna(0).astype(int)
    area["businesses_per_10k"] = np.where(area["residents_est"] > 0, 1e4 * area["active"] / area["residents_est"].clip(lower=1), np.nan)
    area["population_est"] = area["residents_est"]
    wsat = sat.groupby("area").apply(lambda g: np.average(g["saturation"], weights=g["n"]) if g["n"].sum() else 0.0,
                                     include_groups=False)
    area["avg_saturation"] = area["area"].map(wsat).fillna(0)
    area["most_saturated"] = area["area"].map(sat[sat["n"] >= 5].sort_values("saturation").groupby("area").tail(1).set_index("area")["sector"])
    local = sat[sat["local_share"] >= 0.6]
    area["best_opportunity"] = area["area"].map(local.sort_values("saturation").groupby("area").head(1).set_index("area")["sector"])
    area["has_licenses"] = area["active"] > 0
    area_sector = sat.merge(act.groupby(["area", "sector"]).agg(closure_rate_12m=("p_close_12m", "mean"),
                                                                avg_score=("score", "mean")).reset_index(),
                            on=["area", "sector"], how="left")
    area_sector["forecast_closures_12m"] = (area_sector["closure_rate_12m"].fillna(0) * area_sector["n"]).round(1)
    sector = act.groupby("sector").agg(sim_active=("license_id", "size"), avg_score=("score", "mean"),
                                       forecast_closure_rate=("p_close_12m", "mean")).reset_index()
    real_n = sat.groupby("sector")["n"].sum()
    sector["active"] = sector["sector"].map(real_n)
    sector["forecast_closures_12m"] = sector["forecast_closure_rate"] * sector["active"]

    rng = np.random.default_rng(0)
    pts = act.sample(min(3000, len(act)), random_state=0).merge(districts[["area", "lat", "lon"]], on="area")
    pts["lat"] += rng.normal(0, 0.003, len(pts))
    pts["lon"] += rng.normal(0, 0.003, len(pts))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    area.drop(columns=["polygon"]).to_csv(PROCESSED / "area_stats.csv", index=False)
    area_sector.to_csv(PROCESSED / "area_sector.csv", index=False)
    sector.to_csv(PROCESSED / "sector_forecast.csv", index=False)
    sat.to_csv(PROCESSED / "saturation_current.csv", index=False)
    pts[["license_id", "sector", "area", "capital_aed", "employees", "score", "p_close_12m", "lat", "lon"]] \
        .to_csv(PROCESSED / "map_points.csv", index=False)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    df = load_businesses()
    cutoff = df["issue_date"].max().normalize()
    train, val, test = split(df)
    print(f"Split: train {len(train):,} | val {len(val):,} | test {len(test):,}")

    # --- fit on train (saturation as it was at the end of the training period) ---
    sat_train = saturation_table()  # REAL DED license counts per district x sector
    fb = FeatureBuilder().fit(train, sat_train)
    x_tr = fb.transform(train)
    cox = fit_cox(x_tr, train["duration_months"], train["event"])
    metrics, cal = evaluate_on_test(cox, fb, train, test)
    pv = penalty_variant(fb, train, val, test)
    metrics["c_index_penalty_variant"] = pv["c_index_test"]
    metrics["lambda_penalty_variant"] = pv["lambda"]
    ph = ph_check(cox, x_tr, train["duration_months"], train["event"])

    # --- refit on all data for deployment, with today's saturation ---
    sat_now = saturation_table()
    recent = df[df["issue_date"] >= DEPLOY_START].reset_index(drop=True)  # current regime
    fb_all = FeatureBuilder().fit(recent, sat_now)
    x_all = fb_all.transform(recent)
    cox_all = fit_cox(x_all, recent["duration_months"], recent["event"])
    b = Bundle(features=fb_all, cox=cox_all, lam=pv["lambda"], saturation_table=sat_now,
               x_means=x_all.mean(), metrics=metrics, cutoff=cutoff)
    save_bundle(b)
    precompute_tables(b, df)

    # --- reports ---
    gam = cox_all.params_
    metrics["gamma_reputation"] = float(gam["reputation"])
    metrics["gamma_digital"] = float(gam["digital_presence"])
    metrics["gamma_saturation"] = float(gam["saturation"])
    metrics["rating_prior_mean"] = fb_all.rating_mu
    metrics["digital_presence_weights"] = {D_LABELS[c]: round(fb_all.dpi.weights[c], 3) for c in D_COMPONENTS}
    metrics["ph_violations_at_0.01"] = ph.loc[ph["violates_at_0.01"], "covariate"].tolist()
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    pv["grid"].to_csv(REPORTS / "lambda_grid_penalty_variant.csv", index=False)
    cal.to_csv(REPORTS / "calibration_36m.csv", index=False)
    ph.to_csv(REPORTS / "ph_check.csv", index=False)
    coefs = cox_all.summary[["coef", "exp(coef)", "p"]].rename(columns={"exp(coef)": "hazard_ratio", "p": "p_value"})
    coefs.to_csv(REPORTS / "cox_coefficients.csv")
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    plot_calibration(cal, REPORTS / "figures" / "calibration_36m.png")

    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\nHazard ratios (deployed model):")
    print(coefs.round(3).to_string())
    print("\nSaved models/bundle.joblib, reports/, data/processed/")


if __name__ == "__main__":
    main()

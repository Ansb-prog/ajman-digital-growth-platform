"""Step 8 - The Business Survival Score.

    Score(t, x) = 100 * S0(t) ^ exp(beta . x  +  gamma_R * R  +  gamma_D * D  +  gamma_S * Saturation)

- S0(t): baseline survival curve (Cox model), t = horizon in months (default 36)
- beta . x: firm basics (sector, capital, employees, free zone, premises, legal form)
- R: Google reputation, Bayesian-averaged rating (src/features/build_features.py)
- D: Digital Presence Index, learned weights (src/features/build_features.py)
- Saturation: demand-based market saturation of the sector around the area (src/features/saturation.py)
All gammas are estimated with the betas, so each has a hazard ratio and p-value.
Score = 100 x probability of still operating at t.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.load import ROOT
from src.features.build_features import BASELINE_SECTOR, FeatureBuilder
from src.models.cox import survival_at, survival_curve

BUNDLE_PATH = ROOT / "models" / "bundle.joblib"

FEATURE_LABELS = {
    "capital_z": "Starting capital", "employees_z": "Number of employees", "free_zone": "Free zone license",
    "home_based": "Home-based premises", "llc": "LLC legal form", "sector": "Sector",
    "digital_presence": "Digital presence (marketing)", "reputation": "Google reputation (rating x reviews)",
    "saturation": "Market saturation",
}

DEFAULT_BUSINESS = {
    "sector": "retail", "area": "Nuaimeya 1", "capital_aed": 150_000, "employees": 4, "free_zone": False,
    "legal_form": "sole establishment", "premises_type": "shop", "posts_per_month": 4,
    "engagement_rate": 0.03, "instagram_followers": 500, "google_rating": 4.2, "google_review_count": 10,
    "days_since_last_post": 7, "has_website": False,
}


@dataclass
class Bundle:
    features: FeatureBuilder
    cox: object
    lam: float  # kept for reference: the earlier multiplicative-penalty variant (not used in the score)
    saturation_table: pd.DataFrame  # current (active licenses at data cutoff)
    x_means: pd.Series  # mean of the Cox design matrix (reference business)
    metrics: dict
    cutoff: pd.Timestamp


def save_bundle(b: Bundle, path: Path = BUNDLE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(b, path)


def load_bundle(path: Path = BUNDLE_PATH) -> Bundle:
    return joblib.load(path)


def score_frame(b: Bundle, df: pd.DataFrame, horizon: float = 36) -> pd.DataFrame:
    """Vectorized: survival probability, saturation, digital presence and score for many businesses."""
    x = b.features.transform(df)
    s = survival_at(b.cox, x, horizon)
    return pd.DataFrame({"survival_prob": s, "saturation": x["saturation"].values,
                         "digital_presence": x["digital_presence"].values, "reputation": x["reputation"].values,
                         "score": np.clip(100 * s, 0, 100)}, index=df.index)


def contributions(b: Bundle, x_row: pd.DataFrame) -> pd.DataFrame:
    """Effect of each factor on hazard vs an average Ajman business: beta_i * (x_i - mean_i).

    For a Cox model this decomposition is exact (it equals SHAP values in log-hazard space).
    hazard_change_pct = how much that factor alone raises (+) or lowers (-) the closure risk.
    """
    beta = b.cox.params_
    raw = beta * (x_row.iloc[0] - b.x_means)
    grouped: dict[str, float] = {}
    for k, v in raw.items():
        key = "sector" if k.startswith("sector_") else k
        grouped[key] = grouped.get(key, 0.0) + float(v)
    row = x_row.iloc[0]
    binary_names = {"free_zone": ("Free zone license", "Mainland license"),
                    "home_based": ("Home-based premises", "Commercial premises"),
                    "llc": ("LLC legal form", "Non-LLC legal form")}
    names = []
    for k in grouped:
        if k in binary_names:
            names.append(binary_names[k][0] if row[k] >= 0.5 else binary_names[k][1])
        else:
            names.append(FEATURE_LABELS.get(k, k))
    out = pd.DataFrame({"factor": names, "log_hazard": list(grouped.values())})
    out["hazard_change_pct"] = (np.exp(out["log_hazard"]) - 1) * 100
    out["direction"] = np.where(out["log_hazard"] > 0, "raises risk", "lowers risk")
    return out.reindex(out["log_hazard"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def predict_score(business: dict, horizon_months: int = 36, bundle: Bundle | None = None) -> dict:
    """Score one business. Missing fields fall back to DEFAULT_BUSINESS values."""
    b = bundle or load_bundle()
    biz = {**DEFAULT_BUSINESS, **{k: v for k, v in business.items() if v is not None}}
    df = pd.DataFrame([biz])
    x = b.features.transform(df)
    months = np.arange(0, 61)
    curve = survival_curve(b.cox, x, months)
    s_h = float(np.interp(horizon_months, months, curve))
    sat = float(x["saturation"].iloc[0])
    gamma_s = float(b.cox.params_["saturation"])
    score = float(np.clip(100 * s_h, 0, 100))
    return {
        "score": round(score, 1),
        "survival_prob": round(s_h, 3),
        "survival_1y": round(float(curve[12]), 3),
        "survival_3y": round(float(curve[36]), 3),
        "survival_5y": round(float(curve[60]), 3),
        "saturation": round(sat, 3),
        "saturation_risk_pct": round(100 * (np.exp(gamma_s * sat) - 1), 1),  # extra closure risk vs an unsaturated market
        "bayes_rating": round(float(x["reputation"].iloc[0] + b.features.rating_mu), 2),
        "digital_presence": round(float(x["digital_presence"].iloc[0]), 3),
        "digital_breakdown": b.features.dpi.breakdown(df),
        "curve": pd.DataFrame({"month": months, "survival": curve}),
        "contributions": contributions(b, x),
        "horizon_months": horizon_months,
    }


def best_areas(business: dict, bundle: Bundle | None = None, k: int = 3, min_share: float = 0.02) -> pd.DataFrame:
    """Same business placed in every area -> top-k areas by score.

    Only areas that already hold at least `min_share` of the sector's businesses are considered,
    as a proxy for zoning and real viability (e.g. no factories suggested for residential areas).
    """
    b = bundle or load_bundle()
    sat = b.saturation_table
    sec = sat[sat["sector"] == business.get("sector", DEFAULT_BUSINESS["sector"])]
    tot = max(sec["n"].sum(), 1)
    areas = sorted(sec.loc[sec["n"] / tot >= min_share, "area"]) or sorted(sat["area"].unique())
    rows = pd.DataFrame([{**DEFAULT_BUSINESS, **business, "area": a} for a in areas])
    sc = score_frame(b, rows)
    sc["area"] = areas
    return sc.sort_values("score", ascending=False).head(k)[["area", "score", "saturation"]].reset_index(drop=True)


__all__ = ["Bundle", "predict_score", "best_areas", "score_frame", "load_bundle", "BASELINE_SECTOR"]

"""Step 3 - Feature engineering: Reputation (R) and the Digital Presence Index (D).

Two separate signals, because how GOOD a business is and how VISIBLE it is are different things:

Reputation R (Bayesian average rating, like IMDb's top-250 formula):
    bayes = (n * rating + k * mu) / (n + k)        mu = Ajman average rating, k = 10 "virtual" reviews
    R     = bayes - mu
A 1-star rating from 2 reviews barely moves R (could be one angry customer); a 1-star
rating from 300 reviews moves it almost all the way. R enters the Cox model as its own
covariate, so its effect size is learned directly.

Digital Presence D = sum_i w_i * norm(c_i), measured in the first 6 months:
    c_posts      = log(1 + posts_per_month)
    c_audience   = log(1 + followers * engagement_rate)   # engaged audience: real people who interact
    c_followers  = log(1 + instagram_followers)            # raw reach (can be inflated by bought followers)
    c_reviews    = log(1 + n_reviews * (rating - 1) / 4)   # POSITIVE review volume: many reviews only help if they are good
    c_recency    = exp(-days_since_last_post / 30)
    c_website    = has_website
norm() is min-max scaling between the 1st and 99th percentile of the training data, clipped
to 0-1. Weights are LEARNED: a Cox model is fitted on the normalized components (plus
reputation, sector and size controls); each protective coefficient (beta < 0) gives
w_i = -beta_i, rescaled to sum to 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

# Time-based split by licensing date (no random leakage across cohorts).
# Time-based split by licensing date (no random leakage across cohorts). Ajman's exit rate halved after
# 2019 (19% -> 8-10%), so the model is trained on the recent regime and deployed on 2020+ cohorts.
TRAIN_START = pd.Timestamp("2020-01-01")
TRAIN_END = pd.Timestamp("2021-06-30")
VAL_END = pd.Timestamp("2021-12-31")
DEPLOY_START = pd.Timestamp("2020-01-01")

BASELINE_SECTOR = "professional services"
from src.data.real import SECTORS  # noqa: E402  (11 real sectors)
D_COMPONENTS = ["c_posts", "c_audience", "c_followers", "c_reviews", "c_recency", "c_website"]
D_LABELS = {
    "c_posts": "Posting frequency", "c_audience": "Engaged audience (followers x engagement)",
    "c_followers": "Raw follower count", "c_reviews": "Positive review volume", "c_recency": "Posting recency",
    "c_website": "Has website",
}
PRIOR_REVIEWS = 10  # k in the Bayesian average

# Defaults used when a digital signal is unknown (e.g. research agent found nothing).
DIGITAL_DEFAULTS = {
    "posts_per_month": 0, "engagement_rate": 0.0, "instagram_followers": 0,
    "google_rating": 0.0, "google_review_count": 0, "days_since_last_post": 180, "has_website": False,
}


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train: licensed 2020 to mid-2021. Validation: H2 2021. Test: 2022 onwards."""
    tr = df[(df["issue_date"] >= TRAIN_START) & (df["issue_date"] <= TRAIN_END)]
    va = df[(df["issue_date"] > TRAIN_END) & (df["issue_date"] <= VAL_END)]
    te = df[df["issue_date"] > VAL_END]
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def digital_components(df: pd.DataFrame) -> pd.DataFrame:
    """Raw (un-normalized) Digital Presence components."""
    d = df.copy()
    for k, v in DIGITAL_DEFAULTS.items():
        if k not in d:
            d[k] = v
        d[k] = d[k].fillna(v)
    return pd.DataFrame({
        "c_posts": np.log1p(d["posts_per_month"].astype(float)),
        "c_audience": np.log1p(d["instagram_followers"].astype(float) * d["engagement_rate"].astype(float)),
        "c_followers": np.log1p(d["instagram_followers"].astype(float)),
        "c_reviews": np.log1p(d["google_review_count"].astype(float)
                              * ((d["google_rating"].astype(float).where(d["google_review_count"] > 0, 1.0) - 1) / 4).clip(0, 1)),
        "c_recency": np.exp(-d["days_since_last_post"].astype(float) / 30.0),
        "c_website": d["has_website"].astype(float),
    }, index=df.index)


def bayes_rating(df: pd.DataFrame, mu: float, k: float = PRIOR_REVIEWS) -> pd.Series:
    """Bayesian average rating: shrinks ratings with few reviews toward the Ajman mean mu."""
    n = df.get("google_review_count", pd.Series(0, index=df.index)).fillna(0).astype(float)
    r = df.get("google_rating", pd.Series(np.nan, index=df.index)).astype(float)
    r = r.where(n > 0, mu).fillna(mu)
    return (n * r + k * mu) / (n + k)


def _controls(df: pd.DataFrame, mu: float) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["reputation"] = bayes_rating(df, mu) - mu
    x["log_capital"] = np.log(df["capital_aed"].astype(float))
    x["log_employees"] = np.log1p(df["employees"].astype(float))
    for s in SECTORS:
        if s != BASELINE_SECTOR:
            x[f"sector_{s}"] = (df["sector"] == s).astype(float)
    return x


@dataclass
class DigitalPresenceIndex:
    lo: dict = field(default_factory=dict)
    hi: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    betas: dict = field(default_factory=dict)

    def normalize(self, comps: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=comps.index)
        for c in D_COMPONENTS:
            span = (self.hi[c] - self.lo[c]) or 1.0
            out[c] = ((comps[c] - self.lo[c]) / span).clip(0, 1)
        return out

    def fit(self, df: pd.DataFrame, mu: float) -> "DigitalPresenceIndex":
        comps = digital_components(df)
        for c in D_COMPONENTS:
            self.lo[c] = float(comps[c].quantile(0.01))
            self.hi[c] = float(comps[c].quantile(0.99))
        x = pd.concat([self.normalize(comps), _controls(df, mu)], axis=1)
        x["duration_months"] = df["duration_months"].values
        x["event"] = df["event"].values
        cph = CoxPHFitter(penalizer=0.01).fit(x, "duration_months", "event")
        self.betas = {c: float(cph.params_[c]) for c in D_COMPONENTS}
        raw = {c: max(0.0, -b) for c, b in self.betas.items()}
        tot = sum(raw.values())
        self.weights = {c: (v / tot if tot > 0 else 1 / len(raw)) for c, v in raw.items()}
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        n = self.normalize(digital_components(df))
        return sum(self.weights[c] * n[c] for c in D_COMPONENTS).rename("digital_presence")

    def breakdown(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-component normalized value and weighted contribution (for the UI)."""
        n = self.normalize(digital_components(df))
        return pd.DataFrame({
            "component": [D_LABELS[c] for c in D_COMPONENTS],
            "value_0_1": [float(n[c].iloc[0]) for c in D_COMPONENTS],
            "weight": [self.weights[c] for c in D_COMPONENTS],
            "contribution": [self.weights[c] * float(n[c].iloc[0]) for c in D_COMPONENTS],
        })


@dataclass
class FeatureBuilder:
    """Turns raw business rows into the Cox model matrix (fit stats on train only)."""
    dpi: DigitalPresenceIndex = field(default_factory=DigitalPresenceIndex)
    means: dict = field(default_factory=dict)
    stds: dict = field(default_factory=dict)
    rating_mu: float = 4.0
    rep_lo: float = -3.0
    rep_hi: float = 1.0
    saturation_table: pd.DataFrame | None = None

    def set_saturation(self, table: pd.DataFrame) -> "FeatureBuilder":
        """Market saturation table used for the `saturation` covariate (see src/features/saturation.py)."""
        self.saturation_table = table[["area", "sector", "saturation"]].copy()
        return self

    def saturation_of(self, df: pd.DataFrame) -> np.ndarray:
        key = self.saturation_table.set_index(["area", "sector"])["saturation"]
        idx = pd.MultiIndex.from_arrays([df["area"].values, df["sector"].values])
        return key.reindex(idx).fillna(0.0).values

    def fit(self, df: pd.DataFrame, saturation_table: pd.DataFrame | None = None) -> "FeatureBuilder":
        if saturation_table is not None:
            self.set_saturation(saturation_table)
        has = df["google_review_count"].fillna(0) > 0
        w = df.loc[has, "google_review_count"].astype(float)
        self.rating_mu = float(np.average(df.loc[has, "google_rating"], weights=w)) if has.any() else 4.0
        self.dpi.fit(df, self.rating_mu)
        # Clip reputation to the range seen in training so the model never extrapolates
        # (e.g. a 1-star business with 500 reviews is rarer than anything in the data).
        rep = bayes_rating(df, self.rating_mu) - self.rating_mu
        self.rep_lo, self.rep_hi = float(rep.quantile(0.002)), float(rep.quantile(0.998))
        for col, s in {"log_capital": np.log(df["capital_aed"].astype(float)),
                       "log_employees": np.log1p(df["employees"].astype(float))}.items():
            self.means[col], self.stds[col] = float(s.mean()), float(s.std())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x = pd.DataFrame(index=df.index)
        x["capital_z"] = (np.log(df["capital_aed"].astype(float)) - self.means["log_capital"]) / self.stds["log_capital"]
        x["employees_z"] = (np.log1p(df["employees"].astype(float)) - self.means["log_employees"]) / self.stds["log_employees"]
        x["free_zone"] = df["free_zone"].astype(float)
        x["home_based"] = (df["premises_type"] == "home-based").astype(float)
        x["llc"] = (df["legal_form"] == "LLC").astype(float)
        for s in SECTORS:
            if s != BASELINE_SECTOR:
                x[f"sector_{s}"] = (df["sector"] == s).astype(float)
        x["reputation"] = (bayes_rating(df, self.rating_mu) - self.rating_mu).clip(self.rep_lo, self.rep_hi).values
        x["digital_presence"] = self.dpi.transform(df).values
        if self.saturation_table is not None:
            x["saturation"] = self.saturation_of(df)
        return x

"""Step 4 - Cox proportional hazards model and fast survival predictions.

S(t | x) = S0(t) ^ exp(beta . x)    (equivalently exp(-H0(t) * exp(beta . x)))
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test


def fit_cox(x: pd.DataFrame, durations: pd.Series, events: pd.Series, penalizer: float = 0.01) -> CoxPHFitter:
    d = x.copy()
    d["duration_months"] = durations.values
    d["event"] = events.values
    return CoxPHFitter(penalizer=penalizer).fit(d, "duration_months", "event")


def baseline_cumhaz(cph: CoxPHFitter) -> tuple[np.ndarray, np.ndarray]:
    """Baseline cumulative hazard H0 as (times, values), for the centered covariates lifelines uses."""
    bch = cph.baseline_cumulative_hazard_
    return bch.index.values.astype(float), bch.iloc[:, 0].values.astype(float)


def h0_at(times: np.ndarray, values: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """Step-function lookup of H0 at arbitrary times."""
    idx = np.searchsorted(times, np.asarray(t, dtype=float), side="right") - 1
    return np.where(idx >= 0, values[np.clip(idx, 0, None)], 0.0)


def survival_at(cph: CoxPHFitter, x: pd.DataFrame, t: np.ndarray | float) -> np.ndarray:
    """S(t | x) for each row of x. `t` may be a scalar or one time per row."""
    times, vals = baseline_cumhaz(cph)
    ph = cph.predict_partial_hazard(x).values
    return np.exp(-h0_at(times, vals, t) * ph)


def survival_curve(cph: CoxPHFitter, x_row: pd.DataFrame, months: np.ndarray) -> np.ndarray:
    times, vals = baseline_cumhaz(cph)
    ph = float(cph.predict_partial_hazard(x_row).values[0])
    return np.exp(-h0_at(times, vals, months) * ph)


def ph_check(cph: CoxPHFitter, x: pd.DataFrame, durations: pd.Series, events: pd.Series) -> pd.DataFrame:
    """Schoenfeld-residual test of the proportional hazards assumption per covariate."""
    d = x.copy()
    d["duration_months"] = durations.values
    d["event"] = events.values
    r = proportional_hazard_test(cph, d, time_transform="rank")
    out = r.summary[["test_statistic", "p"]].reset_index().rename(columns={"index": "covariate", "p": "p_value"})
    out["violates_at_0.01"] = out["p_value"] < 0.01
    return out

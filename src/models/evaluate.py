"""Step 7 - Evaluation metrics: C-index, IPCW Brier score, calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index


def c_index(durations, scores_higher_is_safer, events) -> float:
    """Concordance: share of comparable pairs where the safer-scored business lasted longer."""
    return float(concordance_index(durations, scores_higher_is_safer, events))


def censoring_km(durations, events) -> KaplanMeierFitter:
    """KM estimate of the censoring distribution G(t), used for IPCW weights."""
    return KaplanMeierFitter().fit(durations, 1 - np.asarray(events))


def brier_ipcw(t: float, durations, events, surv_pred_at_t, g: KaplanMeierFitter) -> float:
    """Time-dependent Brier score at horizon t with inverse-probability-of-censoring weights."""
    d = np.asarray(durations, float)
    e = np.asarray(events, int)
    s = np.asarray(surv_pred_at_t, float)
    g_t = max(float(g.survival_function_at_times(t).iloc[0]), 1e-3)
    g_ti = np.clip(g.survival_function_at_times(np.maximum(d - 1e-6, 0)).values, 1e-3, None)
    died = (d <= t) & (e == 1)
    alive = d > t
    loss = np.where(died, (s ** 2) / g_ti, 0.0) + np.where(alive, ((1 - s) ** 2) / g_t, 0.0)
    return float(loss.mean())


def calibration_table(pred_s: np.ndarray, durations, events, t: float, bins: int = 10) -> pd.DataFrame:
    """Predicted vs KM-observed survival at t, by decile of predicted survival."""
    df = pd.DataFrame({"p": pred_s, "d": np.asarray(durations), "e": np.asarray(events)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    rows = []
    for b, g in df.groupby("bin"):
        km = KaplanMeierFitter().fit(g["d"], g["e"])
        rows.append({"bin": int(b), "n": len(g), "predicted": float(g["p"].mean()),
                     "observed": float(km.survival_function_at_times(t).iloc[0])})
    return pd.DataFrame(rows)

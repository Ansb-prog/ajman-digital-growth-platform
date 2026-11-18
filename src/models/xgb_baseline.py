"""Step 5 - XGBoost baseline: binary "still operating at 36 months".

Rows censored before 36 months have an unknown label and are excluded from this
target only (the Cox model uses them).
"""
from __future__ import annotations

import pandas as pd
from xgboost import XGBClassifier

HORIZON = 36


def label_36m(df: pd.DataFrame) -> pd.Series:
    """1 = survived 36m, 0 = closed within 36m, NaN = unknown (censored earlier)."""
    closed = (df["event"] == 1) & (df["duration_months"] <= HORIZON)
    survived = df["duration_months"] > HORIZON
    y = pd.Series(float("nan"), index=df.index)
    y[closed] = 0.0
    y[survived] = 1.0
    return y


def fit_xgb(x: pd.DataFrame, y: pd.Series, seed: int = 42) -> XGBClassifier:
    m = y.notna()
    clf = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.8,
                        colsample_bytree=0.8, random_state=seed, eval_metric="auc")
    clf.fit(x[m], y[m].astype(int))
    return clf

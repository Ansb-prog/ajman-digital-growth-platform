"""Step 2 - Exploratory survival analysis.

Kaplan-Meier survival curves and log-rank tests by sector, area, posting activity
and licensing cohort. Writes figures to reports/figures/ and tables to reports/.

Run:  python -m src.models.km
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test

from src.data.load import ROOT, load_businesses

REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
HORIZONS = (12, 36, 60)  # months


def add_groupings(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived grouping columns used for comparison."""
    df = df.copy()
    df["posting_activity"] = pd.cut(
        df["posts_per_month"], bins=[-1, 0, 3, 8, 1_000],
        labels=["none (0)", "low (1-3)", "medium (4-8)", "high (9+)"],
    )
    df["cohort"] = pd.cut(
        df["issue_year"], bins=[2009, 2014, 2017, 2019, 2021, 2025],
        labels=["2010-14", "2015-17", "2018-19 (high-exit years)", "2020-21 (COVID)", "2022-25"],
    )
    parents = pd.read_csv(ROOT / "data" / "real" / "districts.csv").set_index("district")["parent"]
    df["parent_area"] = df["area"].map(parents).fillna("Other")
    return df


def km_table(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Survival at each horizon (with 95% CI) and median survival per group."""
    rows = []
    for name, g in df.groupby(group, observed=True):
        kmf = KaplanMeierFitter().fit(g["duration_months"], g["event"])
        ci = kmf.confidence_interval_survival_function_
        row = {"grouping": group, "group": str(name), "n": len(g), "events": int(g["event"].sum())}
        for h in HORIZONS:
            row[f"S({h}m)"] = round(float(kmf.survival_function_at_times(h).iloc[0]), 3)
            idx = ci.index[ci.index <= h].max()
            row[f"S({h}m)_low"] = round(float(ci.loc[idx].iloc[0]), 3)
            row[f"S({h}m)_high"] = round(float(ci.loc[idx].iloc[1]), 3)
        med = kmf.median_survival_time_
        row["median_months"] = None if med == float("inf") else round(float(med), 1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("S(36m)")


def plot_km(df: pd.DataFrame, group: str, title: str, path: Path, max_groups: int = 10) -> None:
    """Plot KM curves with confidence bands for each level of `group`."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    big = df[group].value_counts().head(max_groups).index
    order = df[df[group].isin(big)].groupby(group, observed=True)["event"].mean().sort_values().index
    for name in order:
        g = df[df[group] == name]
        KaplanMeierFitter().fit(g["duration_months"], g["event"], label=f"{name} (n={len(g):,})") \
            .plot_survival_function(ax=ax, ci_alpha=0.12, linewidth=1.8)
    for h in HORIZONS:
        ax.axvline(h, color="grey", lw=0.6, ls=":")
    ax.set(title=title, xlabel="Months since license issued", ylabel="Probability still operating",
           ylim=(0, 1.02), xlim=(0, 120))
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    ax.grid(alpha=0.25)
    fig.text(0.99, 0.01, "Synthetic data", ha="right", fontsize=7, color="grey")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def logrank(df: pd.DataFrame, group: str) -> dict:
    """Multivariate log-rank test: do survival curves differ across groups?"""
    d = df.dropna(subset=[group])
    r = multivariate_logrank_test(d["duration_months"], d[group].astype(str), d["event"])
    return {"grouping": group, "test_statistic": round(r.test_statistic, 2),
            "df": r.degrees_of_freedom, "p_value": r.p_value}


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    df = add_groupings(load_businesses())

    specs = [
        ("sector", "Business survival by sector"),
        ("parent_area", "Business survival by area (largest 10)"),
        ("license_type", "Business survival by license type"),
        ("posting_activity", "Business survival by social media posting (first 6 months)"),
        ("cohort", "Business survival by licensing cohort"),
        ("free_zone", "Business survival: free zone vs mainland"),
    ]
    tables, tests = [], []
    for group, title in specs:
        plot_km(df, group, title, FIGS / f"km_{group}.png")
        tables.append(km_table(df, group))
        tests.append(logrank(df, group))

    summary = pd.concat(tables, ignore_index=True)
    tests_df = pd.DataFrame(tests)
    pair = pairwise_logrank_test(df["duration_months"], df["sector"], df["event"]).summary
    pair = pair[["test_statistic", "p"]].reset_index()
    pair.columns = ["sector_a", "sector_b", "test_statistic", "p_value"]

    summary.to_csv(REPORTS / "km_summary.csv", index=False)
    tests_df.to_csv(REPORTS / "logrank_tests.csv", index=False)
    pair.to_csv(REPORTS / "logrank_pairwise_sector.csv", index=False)

    pd.set_option("display.width", 140)
    cols = ["group", "n", "S(12m)", "S(36m)", "S(60m)", "median_months"]
    for group, _ in specs:
        print(f"\n== {group} ==")
        print(summary[summary["grouping"] == group][cols].to_string(index=False))
    print("\n== Log-rank tests (H0: all groups have the same survival) ==")
    print(tests_df.to_string(index=False))
    ns = pair[pair["p_value"] > 0.05]
    print(f"\nSector pairs NOT significantly different (p>0.05): {len(ns)} of {len(pair)}")
    if len(ns):
        print(ns.to_string(index=False))
    print(f"\nFigures -> {FIGS.relative_to(ROOT)}/, tables -> {REPORTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()

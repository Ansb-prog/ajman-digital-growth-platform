import pandas as pd
import pytest

from src.data.generate_synthetic import CALIB_PATH, CUTOFF, OUT_PATH

REQUIRED = [
    "license_id", "issue_date", "cancel_date", "sector", "area", "license_type", "legal_form", "capital_aed",
    "employees", "free_zone", "premises_type", "has_website", "google_rating", "google_review_count",
    "instagram_followers", "posts_per_month", "engagement_rate", "days_since_last_post", "duration_months", "event",
]


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(OUT_PATH)


def test_columns_present(df):
    assert set(REQUIRED) <= set(df.columns)


def test_dates_consistent(df):
    c = df.dropna(subset=["cancel_date"])
    assert (pd.to_datetime(c["cancel_date"]) >= pd.to_datetime(c["issue_date"])).all()
    assert (pd.to_datetime(c["cancel_date"]) <= CUTOFF + pd.Timedelta(days=31)).all()
    assert df.loc[df["event"] == 0, "cancel_date"].isna().all()


def test_real_districts_and_sectors(df):
    d = pd.read_csv("data/real/districts.csv")
    assert set(df["area"]) <= set(d["district"])
    assert df["sector"].nunique() == 11


def test_calibrated_to_real_exits():
    c = pd.read_csv(CALIB_PATH).dropna(subset=["real_exit_rate"])
    assert (abs(c["sim_exit_rate"] - c["real_exit_rate"]) < 0.01).all()
    assert (abs(c["sim_active"] / c["real_active"] - 1) < 0.02).all()


def test_trade_riskier_than_professional_and_industrial(df):
    r = df.groupby("license_type")["event"].mean()
    assert r["Trade"] > r["Professional"] > r["Industrial"]


def test_more_posting_means_fewer_failures(df):
    hi = df[df["posts_per_month"] >= df["posts_per_month"].quantile(0.75)]["event"].mean()
    lo = df[df["posts_per_month"] <= df["posts_per_month"].quantile(0.25)]["event"].mean()
    assert hi < lo

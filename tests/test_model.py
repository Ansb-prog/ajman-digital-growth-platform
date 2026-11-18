import numpy as np
import pandas as pd
import pytest

from src.features.build_features import FeatureBuilder, split
from src.features.saturation import saturation_table
from src.models.score import best_areas, load_bundle, predict_score
from src.research.digital_agent import parse_report, to_signals


@pytest.fixture(scope="module")
def b():
    return load_bundle()


def test_score_bounds(b):
    for sector in ["F&B", "professional services", "manufacturing"]:
        r = predict_score({"sector": sector, "area": "Rashideya 1"}, 36, b)
        assert 0 <= r["score"] <= 100
        assert 0 <= r["digital_presence"] <= 1


def test_higher_saturation_lower_score(b):
    sat = b.saturation_table
    f = sat[sat["sector"] == "F&B"].sort_values("saturation")
    lo, hi = f.iloc[0]["area"], f.iloc[-1]["area"]
    s_lo = predict_score({"sector": "F&B", "area": lo}, 36, b)
    s_hi = predict_score({"sector": "F&B", "area": hi}, 36, b)
    assert s_hi["score"] < s_lo["score"]  # same business, only area (saturation) differs


def test_more_marketing_higher_score(b):
    base = {"sector": "retail", "area": "Nuaimeya 1"}
    a = predict_score({**base, "posts_per_month": 0, "days_since_last_post": 180, "google_review_count": 0}, 36, b)
    c = predict_score({**base, "posts_per_month": 15, "days_since_last_post": 2, "google_review_count": 80}, 36, b)
    assert c["score"] > a["score"]


def test_curve_monotonic(b):
    cv = predict_score({"sector": "tech"}, 36, b)["curve"]["survival"].values
    assert np.all(np.diff(cv) <= 1e-12) and cv[0] == pytest.approx(1.0, abs=1e-6)


def test_best_areas(b):
    ba = best_areas({"sector": "F&B"}, b, k=3)
    assert len(ba) == 3 and ba["score"].is_monotonic_decreasing


def _market(extra):
    """Balanced market: each sector spread proportionally to residents, plus `extra` businesses."""
    from src.features.saturation import load_areas
    a = load_areas()
    rows = []
    for _, r in a.iterrows():
        k = int(r["population_est"] / 2000)
        for s in ["F&B", "retail", "manufacturing", "trading"]:
            rows.append({"area": r["area"], "sector": s, "n": k})
    rows += [{"area": e["area"], "sector": e["sector"], "n": 1} for e in extra]
    return pd.DataFrame(rows).groupby(["area", "sector"], as_index=False)["n"].sum()


def test_saturation_demand_based():
    small = saturation_table(counts=_market([{"area": "Yasmeen", "sector": "F&B"}] * 150)).set_index(["area", "sector"])
    big = saturation_table(counts=_market([{"area": "Nuaimeya 1", "sector": "F&B"}] * 150)).set_index(["area", "sector"])
    assert small["saturation"].between(0, 1).all()
    assert small.loc[("Yasmeen", "F&B"), "saturation"] > big.loc[("Nuaimeya 1", "F&B"), "saturation"]


def test_industrial_cluster_is_not_penalized_like_local_sector():
    extra = [{"area": "Ajman Industrial 2", "sector": "manufacturing"}] * 300 + \
            [{"area": "Ajman Industrial 2", "sector": "F&B"}] * 300
    t = saturation_table(counts=_market(extra)).set_index(["area", "sector"])
    assert t.loc[("Ajman Industrial 2", "manufacturing"), "saturation"] < t.loc[("Ajman Industrial 2", "F&B"), "saturation"]


def test_rating_matters_more_with_more_reviews(b):
    def gap(n):
        lo = predict_score({"sector": "F&B", "google_rating": 2.0, "google_review_count": n}, 36, b)["score"]
        hi = predict_score({"sector": "F&B", "google_rating": 5.0, "google_review_count": n}, 36, b)["score"]
        return hi - lo
    assert gap(2) > 0 and gap(100) > 2 * gap(2)


def test_engaged_audience_beats_raw_followers(b):
    real = predict_score({"sector": "retail", "instagram_followers": 5000, "engagement_rate": 0.04}, 36, b)["score"]
    bought = predict_score({"sector": "retail", "instagram_followers": 5000, "engagement_rate": 0.002}, 36, b)["score"]
    assert real > bought


def test_best_areas_respects_zoning_proxy(b):
    ba = best_areas({"sector": "manufacturing"}, b, k=3)
    sat = b.saturation_table
    sec = sat[sat["sector"] == "manufacturing"]
    share = sec.set_index("area")["n"] / sec["n"].sum()
    assert (share.reindex(ba["area"]) >= 0.02).all()


def test_digital_weights_sum_to_one(b):
    assert sum(b.features.dpi.weights.values()) == pytest.approx(1.0)


def test_time_split_no_overlap():
    from src.data.load import load_businesses
    tr, va, te = split(load_businesses())
    assert tr["issue_date"].max() < va["issue_date"].min() and va["issue_date"].max() < te["issue_date"].min()


def test_parse_agent_report():
    txt = 'Notes...<json>{"business_found": true, "has_website": true, "google_rating": 4.5, ' \
          '"google_review_count": 120, "instagram_followers": 3000, "posts_per_month_estimate": 10, ' \
          '"days_since_last_post": 2}</json>'
    sig = to_signals(parse_report(txt))
    assert sig["posts_per_month"] == 10 and sig["google_review_count"] == 120

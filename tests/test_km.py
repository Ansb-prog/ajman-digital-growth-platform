from src.data.load import load_businesses
from src.models.km import add_groupings, km_table, logrank


def _df():
    return add_groupings(load_businesses().sample(6000, random_state=0))


def test_km_table_bounds():
    t = km_table(_df(), "sector")
    for h in (12, 36, 60):
        assert t[f"S({h}m)"].between(0, 1).all()
    assert (t["S(12m)"] >= t["S(36m)"]).all() and (t["S(36m)"] >= t["S(60m)"]).all()


def test_sector_logrank_significant():
    assert logrank(_df(), "sector")["p_value"] < 0.01

import json

import pandas as pd

from src.data.real import build_annual, build_districts, key


def test_district_keys_match_license_spellings():
    assert key("Al Heliow 1") == key("Helio1") == "helio1"
    assert key("Jurf Industrial Zone1") == key("Jurf Industrial 1")
    assert key("Masfout  14") == key("Masfout14")


def test_districts_have_polygons_and_licenses():
    d = build_districts()
    assert len(d) == 79
    assert d["licenses_listed"].sum() > 50_000
    assert (d["area_km2"] > 0).all()


def test_annual_stock_flow_identity():
    a = build_annual().dropna(subset=["exits"])
    prev = a["active"] - a["new"] + a["exits"]
    assert (prev.round() > 0).all()
    assert a.loc[a["year"] == 2025, "exit_rate"].round(3).iloc[0] == 0.084


def test_forecast_outputs():
    f = pd.read_csv("data/processed/forecast_annual.csv")
    assert list(f["year"]) == [2026, 2027, 2028]
    assert (f["active_lo"] <= f["active"]).all() and (f["active"] <= f["active_hi"]).all()
    assert (f["exits"] > 0).all()
    bt = json.load(open("reports/forecast_backtest.json"))
    assert bt["active_mape_model"] < 0.05

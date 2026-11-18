"""Interactive Folium maps on the 79 real Ajman district polygons (Municipality open data).

build_map:        multi-layer map (survival score, sector saturation, overall saturation, expected closure rate,
                  new licenses, digital presence) + clustered businesses, with a full DED profile popup
build_metric_map: one metric with a legend (DED insights tab)
"""
from __future__ import annotations

import json
from functools import lru_cache

import branca.colormap as cm
import folium
import numpy as np
import pandas as pd
from folium.plugins import MarkerCluster

from src.data.load import ROOT

RISK_SCALE = ["#c0392b", "#e67e22", "#f1c40f", "#7dbb5a", "#2e8b57"]  # low score -> high score
SAT_COLORS = ["#f3e8fb", "#c39bd3", "#8e44ad", "#5b2c6f"]
CENTER, ZOOM = [25.405, 55.50], 12

METRICS = {
    "sector_saturation": "Saturation of selected sector",
    "avg_saturation": "Overall market saturation",
    "competitors_per_10k": "Competitors per 10k residents (selected sector)",
    "forecast_closure_rate": "Expected closure rate, next 12 months",
    "forecast_closures_12m": "Expected closures, next 12 months",
    "new_licenses_8m": "New licenses Nov 2025 - Jun 2026 (real)",
    "active": "Licenses in DED register (real)",
    "businesses_per_10k": "Licenses per 10k residents",
    "avg_score": "Average survival score",
}


def score_color(score: float, lo: float = 40, hi: float = 90) -> str:
    frac = float(np.clip((score - lo) / max(hi - lo, 1e-9), 0, 0.9999))
    return RISK_SCALE[int(frac * len(RISK_SCALE))]


@lru_cache(maxsize=1)
def polygons() -> dict[str, list]:
    d = pd.read_csv(ROOT / "data" / "real" / "districts.csv")
    return {r.district: json.loads(r.polygon) for r in d.itertuples()}


def _features(area: pd.DataFrame, values: pd.Series, extra: dict | None = None) -> dict:
    polys = polygons()
    feats = []
    for (_, row), v in zip(area.iterrows(), values):
        if row["area"] not in polys:
            continue
        props = {"area": row["area"], "parent": row.get("parent", ""), "value": None if pd.isna(v) else float(v)}
        if extra:
            props.update({k: f(row) for k, f in extra.items()})
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "Polygon", "coordinates": [polys[row["area"]]]}})
    return {"type": "FeatureCollection", "features": feats}


def _popup_html(row: pd.Series, area_sector: pd.DataFrame, sector: str) -> str:
    a = area_sector[area_sector["area"] == row["area"]]
    sel = a[a["sector"] == sector]
    top = a[a["n"] >= 5].sort_values("saturation", ascending=False).head(3)
    rows = "".join(f"<tr><td>{r.sector}</td><td align=right>{r.saturation:.2f}</td><td align=right>{r.n}</td></tr>"
                   for r in top.itertuples())
    s = sel.iloc[0] if len(sel) else None
    sel_html = (f"<b>{sector}</b>: saturation {s.saturation:.2f} · {int(s.n)} licenses · "
                f"{s.ratio:.1f}x Ajman avg per customer<br>" if s is not None else "")
    score = "n/a" if pd.isna(row.get("avg_score")) else f"{row['avg_score']:.1f}"
    return (f"<div style='font:12px sans-serif;min-width:240px'><b style='font-size:14px'>{row['area']}</b> "
            f"<span style='color:#666'>({row.get('parent', '')})</span><br>"
            f"Licenses in DED register (real): <b>{int(row['active']):,}</b> · new Nov-25–Jun-26: {int(row['new_licenses_8m'])}<br>"
            f"Residents (est.): {int(row['population_est']):,}<br>"
            f"Avg survival score: <b>{score}</b> · expected closure rate (12m): {row['forecast_closure_rate']:.1%}<br>"
            f"Overall saturation: <b>{row['avg_saturation']:.2f}</b><hr style='margin:4px 0'>{sel_html}"
            f"<b>Most saturated sectors</b><table style='width:100%'><tr><th align=left>sector</th><th>sat.</th>"
            f"<th>lic.</th></tr>{rows}</table>"
            f"<b>Market gap</b>: {row.get('best_opportunity', '–')}</div>")


def _layer(name: str, area: pd.DataFrame, values: pd.Series, cmap, fmt: str, show: bool,
           popups: dict[str, str] | None = None) -> folium.FeatureGroup:
    fg = folium.FeatureGroup(name, show=show)
    data = _features(area, values, {"label": lambda r: r["area"]})
    for f in data["features"]:
        v = f["properties"]["value"]
        color = "#dddddd" if v is None else cmap(v)
        gj = folium.GeoJson(f, style_function=lambda _x, c=color: {"fillColor": c, "color": "#555", "weight": 0.8,
                                                                     "fillOpacity": 0.72},
                            highlight_function=lambda _x: {"weight": 2.5, "color": "#111"},
                            tooltip=f"{f['properties']['area']}: " + ("n/a" if v is None else fmt.format(v)))
        if popups and f["properties"]["area"] in popups:
            folium.Popup(popups[f["properties"]["area"]], max_width=320).add_to(gj)
        gj.add_to(fg)
    return fg


def build_map(area: pd.DataFrame, points: pd.DataFrame, area_sector: pd.DataFrame, sector: str) -> folium.Map:
    m = folium.Map(location=CENTER, zoom_start=ZOOM, tiles="OpenStreetMap", control_scale=True)
    pops = {r["area"]: _popup_html(r, area_sector, sector) for _, r in area.iterrows()}
    sel = area_sector[area_sector["sector"] == sector].set_index("area")["saturation"]
    sc = area["avg_score"]
    score_cmap = cm.LinearColormap(RISK_SCALE, vmin=float(np.nanpercentile(sc, 5)), vmax=float(np.nanpercentile(sc, 95)),
                                   caption="Average survival score (low = riskier)")
    sat_cmap = cm.LinearColormap(SAT_COLORS, vmin=0, vmax=0.6, caption="Market saturation (0 = under-served)")
    cr = area["forecast_closure_rate"]
    close_cmap = cm.LinearColormap(["#fdf2e9", "#e59866", "#a04000"], vmin=float(cr.min()), vmax=float(cr.max()))
    nl = area["new_licenses_8m"]
    new_cmap = cm.LinearColormap(["#eaf2f8", "#5dade2", "#1b4f72"], vmin=0, vmax=float(nl.quantile(0.95)))
    dig_cmap = cm.LinearColormap(["#eaf2f8", "#2471a3"], vmin=float(area["avg_digital"].min()),
                                 vmax=float(area["avg_digital"].max()))
    _layer("Average survival score", area, sc, score_cmap, "{:.1f}", True, pops).add_to(m)
    _layer(f"Saturation: {sector}", area, area["area"].map(sel), sat_cmap, "{:.2f}", False, pops).add_to(m)
    _layer("Overall market saturation", area, area["avg_saturation"], sat_cmap, "{:.2f}", False, pops).add_to(m)
    _layer("Expected closure rate (12 months)", area, cr, close_cmap, "{:.1%}", False, pops).add_to(m)
    _layer("New licenses Nov-25–Jun-26 (real)", area, nl, new_cmap, "{:.0f}", False, pops).add_to(m)
    _layer("Digital presence", area, area["avg_digital"], dig_cmap, "{:.2f}", False, pops).add_to(m)
    fg = folium.FeatureGroup("Businesses (simulated)", show=False)
    cluster = MarkerCluster().add_to(fg)
    for _, p in points.iterrows():
        folium.CircleMarker([p["lat"], p["lon"]], radius=4, color=score_color(p["score"]), fill=True, fill_opacity=0.9,
                            popup=f"{p['sector']} · {p['area']}<br>Score {p['score']:.0f}<br>"
                                  f"P(close in 12m) {p['p_close_12m']:.0%}").add_to(cluster)
    fg.add_to(m)
    score_cmap.add_to(m)
    sat_cmap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def area_metric(area: pd.DataFrame, area_sector: pd.DataFrame, metric: str, sector: str) -> pd.Series:
    sel = area_sector[area_sector["sector"] == sector].set_index("area")
    if metric == "sector_saturation":
        return area["area"].map(sel["saturation"])
    if metric == "competitors_per_10k":
        return area["area"].map(sel["per_10k_residents"])
    return area[metric]


def build_metric_map(area: pd.DataFrame, area_sector: pd.DataFrame, metric: str, sector: str) -> folium.Map:
    m = folium.Map(location=CENTER, zoom_start=ZOOM, tiles="OpenStreetMap", control_scale=True)
    vals = area_metric(area, area_sector, metric, sector).astype(float)
    label = METRICS[metric] + (f" ({sector})" if metric in ("sector_saturation", "competitors_per_10k") else "")
    lo, hi = float(np.nanpercentile(vals, 2)), float(np.nanpercentile(vals, 98))
    if metric == "avg_score":
        cmap = cm.LinearColormap(RISK_SCALE, vmin=lo, vmax=hi, caption=label)
    elif "saturation" in metric:
        cmap = cm.LinearColormap(SAT_COLORS, vmin=0, vmax=max(0.6, hi), caption=label)
    else:
        cmap = cm.LinearColormap(["#fdf2e9", "#e59866", "#a04000"], vmin=lo, vmax=hi if hi > lo else lo + 1, caption=label)
    fmt = "{:.1%}" if metric == "forecast_closure_rate" else ("{:.2f}" if "saturation" in metric else "{:,.0f}")
    _layer(label, area, vals, cmap, fmt, True).add_to(m)
    cmap.add_to(m)
    return m


def build_growth_map(districts: pd.DataFrame, value_col: str = "growth_2025_2028_pct",
                     label: str = "Forecast growth in active licenses 2025-2028 (%)") -> folium.Map:
    """Diverging map: green = growing faster than Ajman, red = cooling. `districts` needs columns district + value."""
    m = folium.Map(location=CENTER, zoom_start=ZOOM, tiles="OpenStreetMap", control_scale=True)
    d = districts.rename(columns={"district": "area"})
    v = d[value_col].astype(float)
    avg = float(np.average(v, weights=d["active_2025"].clip(lower=1)))
    span = float(np.nanpercentile(np.abs(v - avg), 90)) or 1.0
    cmap = cm.LinearColormap(["#c0392b", "#f7f7f7", "#2e8b57"], vmin=avg - span, vmax=avg + span, caption=label)
    pops = {r["area"]: (f"<div style='font:12px sans-serif'><b>{r['area']}</b> ({r['parent']})<br>"
                        f"Licenses 2025: {r['active_2025']:,.0f} → 2028: {r['active_2028']:,.0f}<br>"
                        f"Growth: <b>{r[value_col]:+.0f}%</b> (Ajman {avg:+.0f}%)<br>"
                        f"New licenses 2026: {r['new_2026']:,.0f} · exits 2026: {r['exits_2026']:,.0f}<br>"
                        f"Outlook: <b>{r['outlook']}</b></div>") for _, r in d.iterrows()}
    _layer(label, d, v.clip(avg - span, avg + span), cmap, "{:+.0f}%", True, pops).add_to(m)
    cmap.add_to(m)
    return m


def build_suggestion_map(opp: pd.DataFrame, current: str | None, top: list[str]) -> folium.Map:
    """Dark map hero: green = suggested districts, white = other options, red = applicant's choice."""
    GREEN = "#3DBE5C"
    DARK_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
    DARK_ATTR = "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors"
    polys = polygons()
    cent = pd.read_csv(ROOT / "data" / "real" / "districts.csv").set_index("district")[["lat", "lon"]]
    m = folium.Map(location=CENTER, zoom_start=12, tiles=None, control_scale=False, zoom_control=True)
    folium.TileLayer(DARK_TILES, attr=DARK_ATTR, name="dark").add_to(m)
    for _, r in opp.iterrows():
        d = r["district"]
        if d not in polys:
            continue
        if d in top:
            style = {"fillColor": GREEN, "color": GREEN, "weight": 2, "fillOpacity": 0.65}
        elif d == current:
            style = {"fillColor": "#E5484D", "color": "#E5484D", "weight": 2, "fillOpacity": 0.55}
        else:
            style = {"fillColor": "#ffffff", "color": "#9CA3AF", "weight": 0.6, "fillOpacity": 0.08}
        tip = (f"<b>{d}</b> ({r['parent']})<br>Opportunity {r['opportunity']:.0f} · survival {r['score']:.0f}<br>"
               f"Competing license-activities within ~2 km: {r['competitors_nearby']:.0f} ({r['competitors_per_10k']:.0f} per 10k residents, est.)<br>"
               f"Forecast growth {r['growth_pct']:+.0f}%")
        folium.GeoJson({"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [polys[d]]}},
                       style_function=lambda _x, st_=style: st_, tooltip=folium.Tooltip(tip)).add_to(m)
    for i, d in enumerate(top, start=1):
        r = opp[opp["district"] == d]
        if len(r) and d in polys:
            r = r.iloc[0]
            folium.Marker([cent.loc[d, "lat"], cent.loc[d, "lon"]], icon=folium.DivIcon(
                icon_size=(30, 30), icon_anchor=(15, 15),
                html=f"<div style='width:28px;height:28px;border-radius:50%;background:{GREEN};color:#fff;"
                     f"font:700 13px Inter,sans-serif;display:grid;place-items:center;border:3px solid #fff;"
                     f"box-shadow:0 2px 6px rgba(0,0,0,.4)'>{i}</div>")).add_to(m)
    return m

"""Market saturation per area and sector: supply of competitors vs local demand.

Real-life ideas built in:
1. Demand, not just counts. A sector is saturated when there are many competitors
   *per customer*, so supply is compared with resident population (and visitors for
   tourism/F&B/retail) instead of with the area's other businesses.
2. Catchment. Customers cross area borders, so supply and demand are smoothed over
   neighbouring areas with a distance kernel K(d) = exp(-d / 2 km). Outlying
   enclaves (Masfout, Manama) are effectively their own markets.
3. Local vs B2B sectors. Restaurants and shops compete for nearby customers;
   manufacturers and traders sell Ajman-wide or abroad, and clustering in an
   industrial zone is an advantage (shared suppliers, labour, logistics). Each
   sector has a "local share" l_s; only that share is judged against local demand,
   the rest gets a mild penalty for extreme concentration.

For area a and sector s:
    supply_as  = sum_b n_bs * K(d_ab)                                # competitors in catchment
    demand_as  = sum_b pop_b * (1 + v_s * visitor_index_b) * K(d_ab)  # customers in catchment
    ratio_as   = (supply_as / demand_as) / (N_s / sum_b pop_b * (1 + v_s * visitor_index_b))
                 -> 1.0 = as many competitors per customer as Ajman on average
    lq_as      = classic location quotient (sector share of area's businesses vs Ajman)
    g(r)       = clip((r - 0.75) / 2.25, 0, 1)    -> 0 below 0.75x average, 1 at 3x
    Saturation = l_s * g(ratio_as) + (1 - l_s) * 0.2 * g(lq_as)
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src.data.load import ROOT

DISTRICTS_PATH = ROOT / "data" / "real" / "districts.csv"
REAL_COUNTS_PATH = ROOT / "data" / "real" / "district_sector.csv"
KERNEL_KM = 2.0
CLUSTER_PENALTY = 0.2

# l_s: share of a sector's customers that are local; v_s: how much visitors add to demand.
SECTOR_MARKET = {
    "F&B":                   {"local": 1.0, "visitor": 1.0},
    "personal services":     {"local": 1.0, "visitor": 0.2},
    "retail":                {"local": 0.9, "visitor": 0.4},
    "tourism":               {"local": 0.8, "visitor": 3.0},
    "auto services":         {"local": 0.6, "visitor": 0.0},
    "real estate":           {"local": 0.6, "visitor": 0.5},
    "professional services": {"local": 0.6, "visitor": 0.0},
    "tech":                  {"local": 0.3, "visitor": 0.0},
    "trading":               {"local": 0.3, "visitor": 0.0},
    "construction":          {"local": 0.3, "visitor": 0.0},
    "manufacturing":         {"local": 0.1, "visitor": 0.0},
}


@lru_cache(maxsize=1)
def load_areas() -> pd.DataFrame:
    """The 79 real Ajman districts (Municipality polygons) with ESTIMATED residents and visitor index."""
    d = pd.read_csv(DISTRICTS_PATH)
    return d.rename(columns={"district": "area", "residents_est": "population_est"})


@lru_cache(maxsize=1)
def real_counts() -> pd.DataFrame:
    """Real DED license-activity counts per district x sector (columns: area, sector, n)."""
    return pd.read_csv(REAL_COUNTS_PATH).rename(columns={"district": "area", "licenses": "n"})


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def kernel_matrix(areas: pd.DataFrame) -> np.ndarray:
    lat, lon = areas["lat"].values, areas["lon"].values
    d = _haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    return np.exp(-d / KERNEL_KM)


def _g(r):
    return np.clip((np.asarray(r, float) - 0.75) / 2.25, 0, 1)


def saturation_table(active: pd.DataFrame | None = None, areas: pd.DataFrame | None = None,
                     counts: pd.DataFrame | None = None) -> pd.DataFrame:
    """Saturation for every area x sector pair.

    Supply comes from `counts` (area, sector, n), e.g. the REAL DED license counts, or is counted
    from `active` business rows. With neither, the real DED counts are used.

    Columns: area, sector, n, per_10k_residents, ratio (competitors per customer vs Ajman avg),
    lq, local_share, saturation (0 = under-served, 1 = heavily saturated).
    """
    areas = (areas if areas is not None else load_areas()).reset_index(drop=True)
    names = areas["area"].tolist()
    sectors = sorted(SECTOR_MARKET)
    K = kernel_matrix(areas)
    pop = areas["population_est"].values.astype(float)
    vis = areas["visitor_index"].values.astype(float)
    if counts is None:
        counts = real_counts() if active is None else active.groupby(["area", "sector"]).size().rename("n").reset_index()
    wide = counts.pivot_table(index="area", columns="sector", values="n", aggfunc="sum", fill_value=0)
    n = wide.reindex(index=names, columns=sectors, fill_value=0).values.astype(float)  # areas x sectors
    pop = np.maximum(pop, 1.0)
    area_tot = n.sum(1, keepdims=True)
    sector_tot = n.sum(0, keepdims=True)
    lq = (n / np.clip(area_tot, 1, None)) / np.clip(sector_tot / max(n.sum(), 1), 1e-9, None)

    rows = []
    for j, s in enumerate(sectors):
        m = SECTOR_MARKET[s]
        demand_base = pop * (1 + m["visitor"] * vis)
        supply = K @ n[:, j]
        demand = K @ demand_base
        avg = n[:, j].sum() / demand_base.sum()
        ratio = (supply / np.clip(demand, 1, None)) / max(avg, 1e-12)
        sat = m["local"] * _g(ratio) + (1 - m["local"]) * CLUSTER_PENALTY * _g(lq[:, j])
        for i, a in enumerate(names):
            rows.append({"area": a, "sector": s, "n": int(n[i, j]),
                         "per_10k_residents": round(1e4 * n[i, j] / pop[i], 2),
                         "ratio": float(ratio[i]), "lq": float(lq[i, j]), "local_share": m["local"],
                         "saturation": float(np.clip(sat[i], 0, 1))})
    return pd.DataFrame(rows)


def active_at(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """Businesses licensed on/before `date` and not cancelled by then."""
    issued = pd.to_datetime(df["issue_date"]) <= date
    cancel = pd.to_datetime(df["cancel_date"])
    return df[issued & (cancel.isna() | (cancel > date))]


def lookup(table: pd.DataFrame, area: str, sector: str) -> float:
    row = table[(table["area"] == area) & (table["sector"] == sector)]
    return float(row["saturation"].iloc[0]) if len(row) else 0.0


def add_saturation(df: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=["saturation"], errors="ignore").merge(
        table[["area", "sector", "saturation"]], on=["area", "sector"], how="left")
    out["saturation"] = out["saturation"].fillna(0.0)
    return out

"""Real Ajman open data -> tidy tables used by the model, maps and forecasts.

Raw inputs (data/real/raw/), pulled from the Ajman Government Open Data portal (data.ajman.ae)
and the files provided by the hackathon organizers:

  district_sector_licenses.txt  DED "Total License Per Activity": license-activity counts per district,
                                activities mapped to 11 sectors with keyword rules (see SECTOR_RULES_DOC)
  districts_poly.txt            Municipality "Districts area": 79 district polygons (simplified) + centroids
  district_misc.txt             DED "License in Ajman (Area)": listed licenses + starts Nov-2025..Jun-2026 per
                                district; DLRR "Completed Real Estate Projects": units per district;
                                DED economic permits (promotions / discounts) by sector
  afz_year_status.txt           Free Zones Authority "Companies by License Type": status by start year
  afz_type_status.txt           same, status by license type
  chamber_2023_by_cohort.txt    Ajman Chamber "Membership Transactions" (2023): per registration year and
                                license type -> members, cancelled in 2023, late renewals, renewals
  ajmanded_annual.xlsx          organizers: GDP, inflation, new/renewed/cancelled/active licenses 2016-2025
  afz_monthly.xlsx              organizers: Ajman Free Zone new company records per month, 2024-01..2026-11

Run:  python -m src.data.real      (writes data/real/*.csv)
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from src.data.load import ROOT

RAW = ROOT / "data" / "real" / "raw"
OUT = ROOT / "data" / "real"

SECTORS = ["F&B", "retail", "trading", "construction", "personal services", "auto services",
           "professional services", "tech", "real estate", "tourism", "manufacturing"]

SECTOR_RULES_DOC = (
    "DED activity names were mapped to sectors with ordered keyword rules: tourism (travel, hotel, ticket...), "
    "real estate, F&B (restaurant, cafeteria, sweets, bakery, juice...), tech (IT, software, e-commerce...), "
    "auto services (vehicle, car, tyre, denting...), manufacturing (manufacturing, factory, printing, packaging...), "
    "construction (contracting, maintenance of buildings, tiling, carpentry, aluminium, A/C installation...), "
    "personal services (hairdressing, beauty, tailoring, laundry, cleaning, repair...), professional services "
    "(consultancy, typing, transactions follow-up, delivery, clinics...), trading (wholesale, import/export, "
    "general trading), retail (retail sale, shops, supermarket, pharmacy...). 0.8% of license-activities matched "
    "no rule ('other') and are excluded."
)

# Parent area for each district family (used for resident estimates and readable labels).
PARENT = [
    ("nuaimeya", "Al Nuaimiya"), ("rashideya", "Al Rashidiya"), ("rumaila", "Al Rumailah"),
    ("nakheel", "Al Nakhil"), ("liwara", "Al Liwara"), ("mushairif", "Al Mushairif"), ("safia", "Al Safia"),
    ("zorah", "Al Zorah"), ("jurfindustrial", "Al Jurf Industrial"), ("jurf", "Al Jurf"),
    ("humaideya", "Al Hamidiya"), ("rawdha", "Al Rawda"), ("mowaihat", "Al Mowaihat"),
    ("ajmanindustrial", "Ajman Industrial"), ("tallah", "Al Tallah"), ("mohammedbinzayed", "Mohammed Bin Zayed"),
    ("ragayeb", "Al Raqaib"), ("aalia", "Al Aaliah"), ("aamra", "Al Amerah"), ("yasmeen", "Al Yasmeen"),
    ("helio", "Al Helio"), ("zahya", "Al Zahya"), ("bahya", "Al Bahia"), ("manama", "Manama"),
    ("masfout", "Masfout"), ("ajmanfreezone", "Ajman Free Zone"),
]

# ESTIMATED residents per parent area (no official district population is published on the portal).
# Replace with Ajman Statistics Center figures when available.
RESIDENTS_EST = {
    "Al Nuaimiya": 120_000, "Al Rashidiya": 80_000, "Al Rumailah": 25_000, "Al Nakhil": 30_000,
    "Al Liwara": 15_000, "Al Mushairif": 15_000, "Al Safia": 3_000, "Al Zorah": 6_000, "Al Jurf": 45_000,
    "Al Jurf Industrial": 25_000, "Al Hamidiya": 45_000, "Al Rawda": 40_000, "Al Mowaihat": 45_000,
    "Ajman Industrial": 50_000, "Al Tallah": 10_000, "Mohammed Bin Zayed": 12_000, "Al Raqaib": 8_000,
    "Al Aaliah": 10_000, "Al Amerah": 25_000, "Al Yasmeen": 15_000, "Al Helio": 15_000, "Al Zahya": 8_000,
    "Al Bahia": 8_000, "Manama": 15_000, "Masfout": 15_000, "Ajman Free Zone": 0,
}
VISITOR_INDEX = {"Al Zorah": 3.0, "Al Nuaimiya": 0.6, "Al Rashidiya": 0.6, "Al Nakhil": 0.6, "Al Rumailah": 0.5,
                 "Masfout": 0.5, "Al Liwara": 0.3, "Al Mushairif": 0.3}

# License-area spellings that differ from the polygon names.
KEY_FIX = {"zahya1": "zahya", "jurf3": "jurf2", "jurfindustrialzone1": "jurfindustrial1",
           "jurfindustrialzone2": "jurfindustrial2", "jurfindustrialzone3": "jurfindustrial3",
           "rashideyatowers": "rashideya3", "nuaimeyatowers": "nuaimeya2", "masfout": "masfout1",
           "manama": "manama8"}


def key(name: str) -> str:
    k = str(name).lower().replace("al heliow", "helio").replace("jurf industrial zone", "jurfindustrial")
    k = re.sub(r"\s+", "", k.replace("masfoot", "masfout"))
    return KEY_FIX.get(k, k)


def parent_of(k: str) -> str:
    for prefix, name in PARENT:
        if k.startswith(prefix):
            return name
    return "Other"


def _polygon_area_km2(coords: list[tuple[float, float]]) -> float:
    lon = np.array([c[0] for c in coords])
    lat = np.array([c[1] for c in coords])
    x = np.radians(lon) * 6371 * np.cos(np.radians(lat.mean()))
    y = np.radians(lat) * 6371
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _kv(s: str) -> dict[str, int]:
    out = {}
    for part in s.split(";"):
        if ":" in part:
            k, v = part.rsplit(":", 1)
            out[key(k)] = out.get(key(k), 0) + int(v)
    return out


def build_districts() -> pd.DataFrame:
    misc = dict(line.split("=", 1) for line in (RAW / "district_misc.txt").read_text().strip().splitlines())
    units, listed, new = _kv(misc["units"]), _kv(misc["listed"]), _kv(misc["new12"])
    rows = []
    for line in (RAW / "districts_poly.txt").read_text().strip().splitlines():
        name, no, city, lat, lon, ring = line.split("|")
        coords = [tuple(float(v) for v in p.split(",")) for p in ring.split()]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        k = key(name)
        nice = re.sub(r"\s+", " ", name.replace("Al Heliow", "Al Helio")).strip()
        rows.append({"district": nice, "key": k, "parent": parent_of(k),
                     "city": {"1": "Ajman", "2": "Manama", "3": "Masfout"}[city],
                     "lat": float(lat), "lon": float(lon), "area_km2": _polygon_area_km2(coords),
                     "polygon": json.dumps([[round(x, 4), round(y, 4)] for x, y in coords]),
                     "re_units": units.get(k, 0), "licenses_listed": listed.get(k, 0),
                     "new_licenses_8m": new.get(k, 0)})
    d = pd.DataFrame(rows)
    # residents: parent estimate split 50% by land area, 50% by registered real-estate units (if any in parent)
    d["residents_est"] = 0.0
    for p, g in d.groupby("parent"):
        tot = RESIDENTS_EST.get(p, 5_000)
        area_share = g["area_km2"] / g["area_km2"].sum()
        unit_share = g["re_units"] / g["re_units"].sum() if g["re_units"].sum() > 0 else area_share
        d.loc[g.index, "residents_est"] = tot * (0.5 * area_share + 0.5 * unit_share)
    d["residents_est"] = d["residents_est"].round().astype(int)
    d["visitor_index"] = d["parent"].map(VISITOR_INDEX).fillna(0.1)
    return d


def build_district_sector(districts: pd.DataFrame) -> pd.DataFrame:
    rows = [r.split("|") for r in (RAW / "district_sector_licenses.txt").read_text().strip().split(";")]
    df = pd.DataFrame(rows, columns=["raw", "sector", "n"])
    df["n"] = df["n"].astype(int)
    df["key"] = df["raw"].map(key)
    total = df["n"].sum()
    known = df[df["key"].isin(districts["key"]) & df["sector"].isin(SECTORS)]
    out = known.groupby(["key", "sector"])["n"].sum().unstack(fill_value=0).reindex(columns=SECTORS, fill_value=0)
    out = out.reindex(districts["key"], fill_value=0)
    out.index = districts["district"].values
    long = out.stack().rename("licenses").reset_index().rename(columns={"level_0": "district"})
    long.columns = ["district", "sector", "licenses"]
    long.attrs["coverage"] = known["n"].sum() / total
    return long


def build_annual() -> pd.DataFrame:
    a = pd.read_excel(RAW / "ajmanded_annual.xlsx").replace("-", np.nan).infer_objects()
    a.columns = ["year", "gdp_aed_m", "inflation_pct", "new", "renewed", "cancelled", "active", "permits"]
    a = a.apply(pd.to_numeric, errors="coerce")
    a["year"] = a["year"].astype(int)
    # exits = everything that left the active stock: formal cancellations + licenses not renewed
    a["exits"] = a["active"].shift(1) + a["new"] - a["active"]
    a["exit_rate"] = a["exits"] / a["active"].shift(1)
    a["renewal_rate"] = a["renewed"] / a["active"].shift(1)
    a["entry_rate"] = a["new"] / a["active"].shift(1)
    a["gdp_growth_pct"] = 100 * a["gdp_aed_m"].pct_change(fill_method=None)
    return a


def build_afz_monthly() -> pd.DataFrame:
    m = pd.read_excel(RAW / "afz_monthly.xlsx")
    keep = ["Month", "Company records"] + [c for c in m.columns if c.startswith(("License:", "Facility:", "Legal form:"))]
    m = m[keep].rename(columns={"Month": "month", "Company records": "new_companies"})
    m["month"] = pd.to_datetime(m["month"])
    m["partial_month"] = False
    m.loc[m.index[-1], "partial_month"] = True  # last month of the export is incomplete
    return m


def build_survival_evidence() -> dict[str, pd.DataFrame]:
    """Real survival signals: AFZ share still active by cohort age; Chamber 2023 cancellation rate by age/type."""
    afz = pd.DataFrame([r.split("|") for r in (RAW / "afz_year_status.txt").read_text().strip().split(";")],
                       columns=["year", "status", "n"])
    afz["n"] = afz["n"].astype(int)
    afz["year"] = afz["year"].astype(int)
    alive = {"Active", "Under Amendment", "Under Renewal"}
    g = afz.assign(alive=afz["status"].isin(alive) * afz["n"]).groupby("year")[["n", "alive"]].sum()
    g["share_active"] = g["alive"] / g["n"]
    g["age_years_at_snapshot"] = 2020.5 - g.index  # newest start date in export: June 2020
    afz_curve = g.reset_index()

    afz_t = pd.DataFrame([r.split("|") for r in (RAW / "afz_type_status.txt").read_text().strip().split(";")],
                         columns=["license_type", "status", "n"])
    afz_t["n"] = afz_t["n"].astype(int)
    t = afz_t.assign(alive=afz_t["status"].isin(alive) * afz_t["n"]).groupby("license_type")[["n", "alive"]].sum()
    t["share_active"] = t["alive"] / t["n"]

    lt = {"تجارية": "Trade", "مهنية": "Professional", "صناعية": "Industrial", "Freelance": "Freelance",
          "رخصة بدايات": "Startup (Bedayat)"}
    ch = []
    for r in (RAW / "chamber_2023_by_cohort.txt").read_text().strip().split(";"):
        y, typ, vals = r.split("|")
        members, cancelled, late, renewals = map(int, vals.split(","))
        if y == "NULL":
            continue
        ch.append({"reg_year": int(y), "license_type": lt.get(typ, typ), "members": members,
                   "cancelled_2023": cancelled, "late_renewal_2023": late})
    ch = pd.DataFrame(ch)
    ch["age_in_2023"] = 2023 - ch["reg_year"]
    return {"afz_cohort": afz_curve, "afz_type": t.reset_index(), "chamber": ch}


def chamber_hazard_by_type(ch: pd.DataFrame) -> pd.DataFrame:
    """2023 formal-cancellation rate by license type and age band (members registered before 2023)."""
    c = ch[(ch["age_in_2023"] >= 1)].copy()
    c["age_band"] = pd.cut(c["age_in_2023"], [0, 1, 3, 6, 100], labels=["1", "2-3", "4-6", "7+"])
    out = c.groupby(["license_type", "age_band"], observed=True)[["members", "cancelled_2023", "late_renewal_2023"]].sum()
    out["cancel_rate"] = out["cancelled_2023"] / out["members"]
    out["late_rate"] = out["late_renewal_2023"] / out["members"]
    return out.reset_index()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = build_districts()
    ds = build_district_sector(d)
    d = d.merge(ds.groupby("district")["licenses"].sum().rename("license_activities"), on="district")
    d.to_csv(OUT / "districts.csv", index=False)
    ds.to_csv(OUT / "district_sector.csv", index=False)
    a = build_annual()
    a.to_csv(OUT / "annual.csv", index=False)
    build_afz_monthly().to_csv(OUT / "afz_monthly.csv", index=False)
    ev = build_survival_evidence()
    for k, v in ev.items():
        v.to_csv(OUT / f"evidence_{k}.csv", index=False)
    hz = chamber_hazard_by_type(ev["chamber"])
    hz.to_csv(OUT / "evidence_chamber_hazard.csv", index=False)

    print(f"Districts: {len(d)} | residents (est.) {d['residents_est'].sum():,} | "
          f"license-activities mapped {ds['licenses'].sum():,} ({ds.attrs['coverage']:.1%} of total)")
    print(ds.groupby("sector")["licenses"].sum().sort_values(ascending=False).to_string())
    print("\nAnnual (real):")
    print(a[["year", "new", "cancelled", "active", "exits", "exit_rate", "renewal_rate"]].round(3).to_string(index=False))
    print("\nAFZ share still active by cohort (snapshot mid-2020):")
    print(ev["afz_cohort"][["year", "n", "share_active"]].round(3).to_string(index=False))
    print("\nChamber 2023 cancellation rate by type/age:")
    print(hz.round(3).to_string(index=False))


if __name__ == "__main__":
    main()


# Python port of the keyword rules used in the browser to map DED activity names to sectors.
_RULES = [
    ("tourism", r"travel|tour|hotel|hostel|resort|touris|ticket|umrah|hajj|holiday|cruise"),
    ("real estate", r"real estate|property|properties|land and real|owners association"),
    ("F&B", r"restaurant|cafeteria|caf[eé]|coffee|beverage|sweet|bakery|bread|juice|sandwich|pastr|catering|kitchen|grill|shawarma|food prepar|meals|ice cream|popcorn|kiln|confection|cooking|dishes|barbeque|frying|toasting|appetizer|salat"),
    ("tech", r"information technology|software|computer|network|web|digital|e-commerce|ecommerce|internet|data|cyber|app development|programming|electronic games|it "),
    ("auto services", r"vehicle|car |cars |cars$|car$|automobile|tyre|tire|motor|garage|denting|oil change|auto |boat repair|buses rental"),
    ("manufacturing", r"manufactur|factory|production of|industry|industrial|fabricat|assembl|printing|recycl|metal and iron|blacksmith|foundry|plastic products|mixing|mill|packaging|coating|silk-screen|roasting"),
    ("construction", r"contracting|construction|building projects|buildings maintenance|maintenance of building|plaster|tiling|painting works|tinging|carpentry|aluminium|aluminum|glass|electromechanical|electrical fittings|plumbing|sanitary|insulation|d[eé]cor|interior|swimming pool|landscap|air-?condition|scaffold|steel works|gypsum|marble|curtain|cladding|demolition|excavat|paving|fencing|elevator|lift|doors|windows|block works|stone installation|wallpaper|ceiling|fire fighting|installation|water fittings|water pumps"),
    ("personal services", r"hairdress|beauty|personal care|salon|tailor|sewing|knitting|ironing|washing|laundry|dry clean|henna|massage|bath|barber|cleaning|embroider|shoe repair|repair|photograph|nursery|nurseries|day care|fitness|gym|sport|tutor|training|education|wedding|event|billiard|snooker|club|manicure|pedicure|wig|key cutting|renting"),
    ("professional services", r"consult|accounting|audit|legal|advocate|lawyer|law offices|administrative|businessmen|business center|typing|photocopy|transactions|clearing|recruit|manpower|management|marketing|advertis|design|engineering|survey|translation|insurance|brokerage|broker|delivery|courier|logistic|shipping|freight|transport|security|clinic|medical|dental|health|laborator|veterinar|services|debts|exchange"),
    ("trading", r"wholesale|trading|import|export|general trading|distribut|supply|suppliers|commission agent|agent"),
    ("retail", r"retai|retail|sale|sell|shop|store|super ?market|mini market|grocery|pharmacy|optic|jewel|perfume|boutique|mall|bookshop|stationery|florist|flowers|mobile|gift|toys"),
]


def classify_activity(name: str) -> str:
    s = str(name).lower()
    for sector, pat in _RULES:
        if re.search(pat, s):
            return sector
    return "other"


# Top DED activities by number of licenses (DED "Total License Per Activity"), for the mapping check.
TOP_ACTIVITIES = [
    ("Buildings Maintenance", 1255), ("Buildings Cleaning Services", 747), ("Cafeteria", 604),
    ("Retai sale of Women Ready-made Clothes", 586), ("Restaurant", 576), ("Importing", 487), ("Exporting", 455),
    ("Tailoring and Sewing Women Arab Clothes (Women Abayas)", 454), ("All Kind Building Projects Contracting", 451),
    ("Men Hairdressing, Trimming and Styling", 424), ("General Trading", 399), ("Women Personal Care and Beauty", 360),
    ("Décor Works", 300), ("Real Estate Purchase and Sale Brokerage", 299), ("Retai sale of Perfume", 266),
    ("Administrative Services - Businessmen Services", 263), ("Typing and Documents Photocopying Services", 237),
    ("Super Market", 225), ("Vehicles Mechanics Repair", 189), ("Sell Mobile Phones - Retail", 171),
    ("Delivery Services Coordination & Provision", 158), ("Pharmacy", 147), ("Travel Tickets Selling", 127),
    ("Information Technology Network Services", 105), ("Cars Rental", 117),
]
